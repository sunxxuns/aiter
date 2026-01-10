#!/usr/bin/env python3
"""
Rigorous test for Integration Step 3: Full FP8 Flash Attention (head_dim=32)
Tests O = softmax(Q @ K^T) @ V
"""

import torch
import numpy as np
import ctypes
import struct
import subprocess
import sys

# Initialize HIP
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

def build_kernel():
    result = subprocess.run(
        ['/opt/rocm/llvm/bin/clang++', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
         '-mcpu=gfx950', '-o', 'integrate_step3_hd32.co', 'integrate_step3_hd32.s'],
        cwd='/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8',
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    return True

def load_kernel(filename, symbol):
    module = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), filename.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to load {filename}: error {ret}")
    func = ctypes.c_void_p()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to get function {symbol}: error {ret}")
    return module, func

def launch_kernel(func, args_bytes, shared_mem=6144):
    args_gpu = torch.from_numpy(np.frombuffer(args_bytes, dtype=np.uint8)).cuda()
    kernarg_ptr = ctypes.c_void_p(args_gpu.data_ptr())
    kernarg_size = ctypes.c_size_t(len(args_bytes))
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
        HIP_LAUNCH_PARAM_END
    )
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, shared_mem, None, None, extra)
    hip.hipDeviceSynchronize()

def to_fp8(x):
    return x.to(torch.float8_e4m3fn).view(torch.uint8)

def from_fp8(x):
    return x.view(torch.float8_e4m3fn).float()

def run_test(name, Q_f32, K_f32, V_f32, func, atol=0.15):
    """Run a single test case"""
    print(f"\n  Test: {name}")
    
    # Convert to FP8 and back for reference
    Q_fp8 = to_fp8(Q_f32)
    K_fp8 = to_fp8(K_f32)
    V_fp8 = to_fp8(V_f32)
    Q_quant = from_fp8(Q_fp8)
    K_quant = from_fp8(K_fp8)
    V_quant = from_fp8(V_fp8)
    
    # Reference: O = softmax(Q @ K^T) @ V
    S_ref = Q_quant @ K_quant.T
    P_ref = torch.softmax(S_ref, dim=-1)
    O_ref = P_ref @ V_quant
    
    # Allocate GPU buffers
    Q_gpu = Q_fp8.cuda()
    K_gpu = K_fp8.cuda()
    V_gpu = V_fp8.cuda()
    O_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    # Launch kernel
    args = bytearray(32)
    struct.pack_into('Q', args, 0, O_out.data_ptr())
    struct.pack_into('Q', args, 8, Q_gpu.data_ptr())
    struct.pack_into('Q', args, 16, K_gpu.data_ptr())
    struct.pack_into('Q', args, 24, V_gpu.data_ptr())
    
    launch_kernel(func, args)
    
    O_kernel = O_out.cpu()
    
    # Compare
    diff = (O_kernel - O_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    print(f"    O_ref[0,0:4] = {O_ref[0,0:4].tolist()}")
    print(f"    O_kernel[0,0:4] = {O_kernel[0,0:4].tolist()}")
    print(f"    O_ref mean = {O_ref.mean().item():.4f}")
    print(f"    O_kernel mean = {O_kernel.mean().item():.4f}")
    print(f"    Max abs error: {max_err:.4f}")
    print(f"    Mean abs error: {mean_err:.4f}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(O_kernel).any().item()
    has_inf = torch.isinf(O_kernel).any().item()
    
    if has_nan:
        print(f"    ERROR: Output contains NaN!")
        nan_count = torch.isnan(O_kernel).sum().item()
        print(f"    NaN count: {nan_count}/1024")
        return False
    if has_inf:
        print(f"    ERROR: Output contains Inf!")
        return False
    
    # Check relative error for significant values
    mask = O_ref.abs() > 0.01
    if mask.any():
        rel_err = (diff[mask] / O_ref[mask].abs()).max().item()
        print(f"    Max rel error: {rel_err:.4f}")
    
    passed = max_err < atol
    if passed:
        print(f"    ✓ PASSED")
    else:
        print(f"    ✗ FAILED")
        print(f"    Error distribution by quadrant:")
        print(f"      rows 0-7: max={diff[0:8,:].max():.4f}")
        print(f"      rows 8-15: max={diff[8:16,:].max():.4f}")
        print(f"      rows 16-23: max={diff[16:24,:].max():.4f}")
        print(f"      rows 24-31: max={diff[24:32,:].max():.4f}")
    
    return passed

def main():
    print("=" * 70)
    print("Integration Step 3: Full FP8 Flash Attention (head_dim=32)")
    print("=" * 70)
    
    # Build kernel
    print("\nBuilding kernel...")
    if not build_kernel():
        return 1
    print("Build successful!")
    
    # Load kernel
    module, func = load_kernel(
        '/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step3_hd32.co',
        '_ZN5aiter20integrate_step3_hd32E'
    )
    
    results = []
    
    # Test 1: Uniform Q=K=V=1
    print("\n" + "-" * 50)
    Q = torch.ones(32, 32)
    K = torch.ones(32, 32)
    V = torch.ones(32, 32)
    results.append(run_test("Uniform Q=K=V=1 (expect O=1)", Q, K, V, func))
    
    # Test 2: Random small values
    print("\n" + "-" * 50)
    torch.manual_seed(42)
    Q = torch.randn(32, 32) * 0.3
    K = torch.randn(32, 32) * 0.3
    V = torch.randn(32, 32) * 0.3
    results.append(run_test("Random small values", Q, K, V, func))
    
    # Test 3: Random medium values
    print("\n" + "-" * 50)
    torch.manual_seed(123)
    Q = torch.randn(32, 32) * 0.5
    K = torch.randn(32, 32) * 0.5
    V = torch.randn(32, 32) * 0.5
    results.append(run_test("Random medium values", Q, K, V, func))
    
    # Test 4: Positive V values
    print("\n" + "-" * 50)
    torch.manual_seed(456)
    Q = torch.randn(32, 32) * 0.3
    K = torch.randn(32, 32) * 0.3
    V = torch.rand(32, 32) * 0.5 + 0.25  # All positive
    results.append(run_test("Random Q,K with positive V", Q, K, V, func))
    
    # Test 5: V = eye (identity-like)
    print("\n" + "-" * 50)
    Q = torch.ones(32, 32) * 0.5
    K = torch.ones(32, 32) * 0.5
    V = torch.eye(32)
    results.append(run_test("V=identity (selective attention)", Q, K, V, func))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

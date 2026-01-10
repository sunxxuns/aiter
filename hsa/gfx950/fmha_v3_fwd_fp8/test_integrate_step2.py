#!/usr/bin/env python3
"""
Rigorous test for Integration Step 2: QK MFMA + Softmax
Tests P = softmax(Q @ K^T) with various input patterns
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
    """Build the kernel"""
    result = subprocess.run(
        ['/opt/rocm/llvm/bin/clang++', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
         '-mcpu=gfx950', '-o', 'integrate_step2_softmax.co', 'integrate_step2_softmax.s'],
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

def launch_kernel(func, args_bytes, shared_mem=4096):
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

def run_test(name, Q_f32, K_f32, func, atol=0.01, rtol=0.05):
    """Run a single test case"""
    print(f"\n  Test: {name}")
    
    # Convert to FP8 and back for reference
    Q_fp8 = to_fp8(Q_f32)
    K_fp8 = to_fp8(K_f32)
    Q_quant = from_fp8(Q_fp8)
    K_quant = from_fp8(K_fp8)
    
    # Reference: P = softmax(Q @ K^T, dim=-1)
    S_ref = Q_quant @ K_quant.T
    P_ref = torch.softmax(S_ref, dim=-1)
    
    # Allocate GPU buffers
    Q_gpu = Q_fp8.cuda()
    K_gpu = K_fp8.cuda()
    P_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    # Launch kernel
    args = bytearray(24)
    struct.pack_into('Q', args, 0, P_out.data_ptr())
    struct.pack_into('Q', args, 8, Q_gpu.data_ptr())
    struct.pack_into('Q', args, 16, K_gpu.data_ptr())
    
    launch_kernel(func, args)
    
    P_kernel = P_out.cpu()
    
    # Compare
    diff = (P_kernel - P_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    # Check row sums (should be 1.0)
    row_sums = P_kernel.sum(dim=1)
    row_sum_err = (row_sums - 1.0).abs().max().item()
    
    # Check non-negative
    min_val = P_kernel.min().item()
    
    print(f"    S_ref[0,0:4] = {S_ref[0,0:4].tolist()}")
    print(f"    P_ref[0,0:4] = {P_ref[0,0:4].tolist()}")
    print(f"    P_kernel[0,0:4] = {P_kernel[0,0:4].tolist()}")
    print(f"    Max abs error: {max_err:.6f}")
    print(f"    Mean abs error: {mean_err:.6f}")
    print(f"    Row sum error: {row_sum_err:.6f}")
    print(f"    Min P value: {min_val:.6f}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(P_kernel).any().item()
    has_inf = torch.isinf(P_kernel).any().item()
    
    if has_nan:
        print(f"    ERROR: Output contains NaN!")
        nan_rows = torch.isnan(P_kernel).any(dim=1)
        print(f"    NaN in rows: {nan_rows.nonzero().squeeze().tolist()}")
        return False
    if has_inf:
        print(f"    ERROR: Output contains Inf!")
        return False
    if min_val < 0:
        print(f"    ERROR: Output contains negative values!")
        return False
    if row_sum_err > 0.01:
        print(f"    ERROR: Row sums deviate significantly from 1.0!")
        print(f"    Row sums: {row_sums[:8].tolist()}")
        return False
    
    passed = max_err < atol
    if passed:
        print(f"    ✓ PASSED")
    else:
        print(f"    ✗ FAILED")
        # Show error distribution
        print(f"    Error by row quadrant:")
        print(f"      rows 0-7: max={diff[0:8,:].max():.6f}")
        print(f"      rows 8-15: max={diff[8:16,:].max():.6f}")
        print(f"      rows 16-23: max={diff[16:24,:].max():.6f}")
        print(f"      rows 24-31: max={diff[24:32,:].max():.6f}")
    
    return passed

def main():
    print("=" * 70)
    print("Integration Step 2: QK MFMA + Softmax Test")
    print("=" * 70)
    
    # Build kernel
    print("\nBuilding kernel...")
    if not build_kernel():
        return 1
    print("Build successful!")
    
    # Load kernel
    module, func = load_kernel(
        '/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step2_softmax.co',
        '_ZN5aiter20integrate_step2_softmaxE'
    )
    
    results = []
    
    # Test 1: Uniform Q=K=1
    print("\n" + "-" * 50)
    Q = torch.ones(32, 16)
    K = torch.ones(32, 16)
    results.append(run_test("Uniform Q=K=1 (expect P=1/32)", Q, K, func))
    
    # Test 2: Random small values
    print("\n" + "-" * 50)
    torch.manual_seed(42)
    Q = torch.randn(32, 16) * 0.3
    K = torch.randn(32, 16) * 0.3
    results.append(run_test("Random small values", Q, K, func))
    
    # Test 3: Random medium values
    print("\n" + "-" * 50)
    torch.manual_seed(123)
    Q = torch.randn(32, 16) * 0.5
    K = torch.randn(32, 16) * 0.5
    results.append(run_test("Random medium values", Q, K, func))
    
    # Test 4: One-hot pattern (should have peaked softmax)
    print("\n" + "-" * 50)
    Q = torch.zeros(32, 16)
    K = torch.zeros(32, 16)
    for i in range(32):
        Q[i, i % 16] = 1.0
        K[i, i % 16] = 1.0
    results.append(run_test("One-hot pattern (peaked softmax)", Q, K, func))
    
    # Test 5: Scaled random (larger S values)
    print("\n" + "-" * 50)
    torch.manual_seed(456)
    Q = torch.randn(32, 16) * 0.7
    K = torch.randn(32, 16) * 0.7
    results.append(run_test("Scaled random (larger S)", Q, K, func))
    
    # Test 6: Small positive values
    print("\n" + "-" * 50)
    Q = torch.rand(32, 16) * 0.5
    K = torch.rand(32, 16) * 0.5
    results.append(run_test("Small positive values", Q, K, func))
    
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

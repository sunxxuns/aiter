#!/usr/bin/env python3
"""
Rigorous test for Integration Step 1: QK MFMA
Tests S = Q @ K^T with various input patterns
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
         '-mcpu=gfx950', '-o', 'integrate_step1_qk.co', 'integrate_step1_qk.s'],
        cwd='/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8',
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    return True

def load_kernel(filename, symbol):
    """Load a kernel from .co file"""
    module = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), filename.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to load {filename}: error {ret}")
    func = ctypes.c_void_p()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to get function {symbol}: error {ret}")
    return module, func

def launch_kernel(func, args_bytes):
    """Launch kernel with 1 workgroup of 64 threads"""
    args_gpu = torch.from_numpy(np.frombuffer(args_bytes, dtype=np.uint8)).cuda()
    kernarg_ptr = ctypes.c_void_p(args_gpu.data_ptr())
    kernarg_size = ctypes.c_size_t(len(args_bytes))
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
        HIP_LAUNCH_PARAM_END
    )
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 0, None, None, extra)
    hip.hipDeviceSynchronize()

def to_fp8(x):
    """Convert to FP8 e4m3fn"""
    return x.to(torch.float8_e4m3fn).view(torch.uint8)

def from_fp8(x):
    """Convert FP8 bytes back to float"""
    return x.view(torch.float8_e4m3fn).float()

def run_test(name, Q_f32, K_f32, func, atol=0.1, rtol=0.05):
    """Run a single test case"""
    print(f"\n  Test: {name}")
    
    # Convert to FP8 and back for reference
    Q_fp8 = to_fp8(Q_f32)
    K_fp8 = to_fp8(K_f32)
    Q_quant = from_fp8(Q_fp8)
    K_quant = from_fp8(K_fp8)
    
    # Reference: S = Q @ K^T
    S_ref = Q_quant @ K_quant.T
    
    # Allocate GPU buffers
    Q_gpu = Q_fp8.cuda()
    K_gpu = K_fp8.cuda()
    S_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    # Launch kernel
    args = bytearray(24)
    struct.pack_into('Q', args, 0, S_out.data_ptr())
    struct.pack_into('Q', args, 8, Q_gpu.data_ptr())
    struct.pack_into('Q', args, 16, K_gpu.data_ptr())
    
    launch_kernel(func, args)
    
    S_kernel = S_out.cpu()
    
    # Compare
    diff = (S_kernel - S_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    # Check relative error where S_ref is significant
    mask = S_ref.abs() > 0.1
    if mask.any():
        rel_err = (diff[mask] / S_ref[mask].abs()).max().item()
    else:
        rel_err = 0
    
    print(f"    S_ref[0,0:4] = {S_ref[0,0:4].tolist()}")
    print(f"    S_kernel[0,0:4] = {S_kernel[0,0:4].tolist()}")
    print(f"    Max abs error: {max_err:.6f}")
    print(f"    Mean abs error: {mean_err:.6f}")
    print(f"    Max rel error: {rel_err:.4f}")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(S_kernel).any().item()
    has_inf = torch.isinf(S_kernel).any().item()
    
    if has_nan:
        print(f"    ERROR: Output contains NaN!")
        return False
    if has_inf:
        print(f"    ERROR: Output contains Inf!")
        return False
    
    passed = max_err < atol and rel_err < rtol
    if passed:
        print(f"    ✓ PASSED")
    else:
        print(f"    ✗ FAILED")
        # Show error distribution
        print(f"    Error distribution:")
        print(f"      [0:8,0:8] max: {diff[0:8,0:8].max():.4f}")
        print(f"      [0:8,8:16] max: {diff[0:8,8:16].max():.4f}")
        print(f"      [8:16,0:8] max: {diff[8:16,0:8].max():.4f}")
        print(f"      [24:32,24:32] max: {diff[24:32,24:32].max():.4f}")
    
    return passed

def main():
    print("=" * 70)
    print("Integration Step 1: QK MFMA Test")
    print("=" * 70)
    
    # Build kernel
    print("\nBuilding kernel...")
    if not build_kernel():
        return 1
    print("Build successful!")
    
    # Load kernel
    module, func = load_kernel(
        '/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step1_qk.co',
        '_ZN5aiter17integrate_step1_qkE'
    )
    
    results = []
    
    # Test 1: Uniform Q=K=1
    print("\n" + "-" * 50)
    Q = torch.ones(32, 16)
    K = torch.ones(32, 16)
    results.append(run_test("Uniform Q=K=1 (expect S=16)", Q, K, func))
    
    # Test 2: Identity-like (Q=K=I stretched)
    print("\n" + "-" * 50)
    Q = torch.zeros(32, 16)
    K = torch.zeros(32, 16)
    for i in range(32):
        if i < 16:
            Q[i, i] = 1.0
            K[i, i] = 1.0
    results.append(run_test("Identity pattern", Q, K, func))
    
    # Test 3: Random small values
    print("\n" + "-" * 50)
    torch.manual_seed(42)
    Q = torch.randn(32, 16) * 0.3
    K = torch.randn(32, 16) * 0.3
    results.append(run_test("Random small values", Q, K, func))
    
    # Test 4: Random with larger range
    print("\n" + "-" * 50)
    torch.manual_seed(123)
    Q = torch.randn(32, 16) * 0.5
    K = torch.randn(32, 16) * 0.5
    results.append(run_test("Random medium values", Q, K, func))
    
    # Test 5: Orthogonal-ish patterns
    print("\n" + "-" * 50)
    Q = torch.zeros(32, 16)
    K = torch.zeros(32, 16)
    for i in range(32):
        Q[i, i % 16] = 1.0
        K[i, (i + 8) % 16] = 1.0  # Offset to get some zeros
    results.append(run_test("Orthogonal pattern", Q, K, func))
    
    # Test 6: Structured increasing values
    print("\n" + "-" * 50)
    Q = torch.arange(32*16).reshape(32, 16).float() / 512  # Scale to avoid overflow
    K = torch.arange(32*16).reshape(32, 16).float() / 512
    results.append(run_test("Structured increasing", Q, K, func))
    
    # Test 7: Check specific positions
    print("\n" + "-" * 50)
    Q = torch.zeros(32, 16)
    K = torch.zeros(32, 16)
    Q[0, 0] = 1.0
    K[0, 0] = 1.0
    results.append(run_test("Single element Q[0,0]=K[0,0]=1", Q, K, func))
    
    # Test 8: Check last position
    print("\n" + "-" * 50)
    Q = torch.zeros(32, 16)
    K = torch.zeros(32, 16)
    Q[31, 15] = 1.0
    K[31, 15] = 1.0
    results.append(run_test("Single element Q[31,15]=K[31,15]=1", Q, K, func))
    
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

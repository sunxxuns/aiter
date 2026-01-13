#!/usr/bin/env python3
"""Test FP8 QK kernel with stride-132 LDS layout for zero bank conflicts"""

import torch
import subprocess
import ctypes
import numpy as np
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'qk_fp8_stride132.s', '-o', 'qk_fp8_stride132.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'qk_fp8_stride132.co', 'qk_fp8_stride132.o'], cwd=cwd)
    return str(cwd / 'qk_fp8_stride132.co')

def run_kernel(hip, func, K, Q):
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    return out.cpu().numpy()

def compute_reference(K, Q):
    """Compute S^T = K @ Q^T for full HD=128"""
    K_f32 = K.to(torch.float32)  # 32×128
    Q_f32 = Q.to(torch.float32)  # 32×128
    return (K_f32 @ Q_f32.T).cpu().numpy()  # 32×32

def main():
    print("FP8 QK Kernel with Stride-132 (Zero Bank Conflicts)")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter15qk_fp8_stride132E")
    
    all_pass = True
    
    # Test 1: Uniform (K=Q=1)
    print("\nTest 1: Uniform K = Q = 1.0")
    print("-" * 50)
    K = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    out = run_kernel(hip, func, K, Q)
    ref = compute_reference(K, Q)
    
    # Expected: S^T[i,j] = sum_k(1*1) = 128
    expected = 128.0
    print(f"  Expected: {expected}")
    print(f"  Actual mean: {out.mean():.2f}")
    print(f"  Actual first 4: {out[0]:.2f}, {out[1]:.2f}, {out[2]:.2f}, {out[3]:.2f}")
    
    uniform_pass = np.allclose(out, expected, rtol=0.01)
    print(f"  PASS: {uniform_pass}")
    all_pass &= uniform_pass
    
    # Test 2: Random
    print("\nTest 2: Random K and Q")
    print("-" * 50)
    torch.manual_seed(42)
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    
    out = run_kernel(hip, func, K, Q)
    ref = compute_reference(K, Q)
    
    print(f"  Reference: mean={ref.mean():.3f}, std={ref.std():.3f}")
    print(f"  Actual:    mean={out.mean():.3f}, std={out.std():.3f}")
    
    random_pass = abs(out.mean() - ref.mean()) < 1.0 and abs(out.std() - ref.std()) < 1.0
    print(f"  PASS: {random_pass}")
    all_pass &= random_pass
    
    # Test 3: Single row test
    print("\nTest 3: K[0,:] = 1, rest = 0, Q = 1")
    print("-" * 50)
    K = torch.zeros(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K[0, :] = torch.ones(128, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    out = run_kernel(hip, func, K, Q)
    ref = compute_reference(K, Q)
    
    # S^T[0, :] = 128, rest = 0
    print(f"  Expected sum: {ref.sum():.0f}")
    print(f"  Actual sum: {out.sum():.0f}")
    
    single_pass = abs(out.sum() - ref.sum()) < 10
    print(f"  PASS: {single_pass}")
    all_pass &= single_pass
    
    # Test 4: Multiple seeds
    print("\nTest 4: Multiple random seeds")
    print("-" * 50)
    for seed in [123, 456, 789]:
        torch.manual_seed(seed)
        K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        
        out = run_kernel(hip, func, K, Q)
        ref = compute_reference(K, Q)
        
        mean_diff = abs(out.mean() - ref.mean())
        std_diff = abs(out.std() - ref.std())
        ok = mean_diff < 1.0 and std_diff < 1.0
        print(f"  Seed {seed}: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f} - {'PASS' if ok else 'FAIL'}")
        all_pass &= ok
    
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)
    print(f"OVERALL: {'✅ ALL TESTS PASS' if all_pass else '❌ SOME TESTS FAILED'}")
    print("=" * 70)

if __name__ == "__main__":
    main()

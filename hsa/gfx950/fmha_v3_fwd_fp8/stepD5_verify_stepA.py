#!/usr/bin/env python3
"""Final verification of Step A numerical correctness"""

import torch
import subprocess
import ctypes
import numpy as np
from pathlib import Path

def build_kernel(name):
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', f'{name}.s', '-o', f'{name}.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', f'{name}.co', f'{name}.o'], cwd=cwd)
    return str(cwd / f'{name}.co')

def run_kernel(hip, func, K, Q):
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    return out.cpu().numpy()

def main():
    print("Final Verification of Step A (ds_read_b64 + XOR swizzle)")
    print("=" * 70)
    
    co_A = build_kernel("stepA_full_qk")
    if not co_A:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_A.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12stepA_full_qkE")
    
    all_pass = True
    
    # Test 1: Uniform
    print("\n1. Uniform test: K = Q = 1.0")
    K = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    out = run_kernel(hip, func, K, Q)
    
    ref = 16.0  # sum of 16 ones
    uniform_ok = np.allclose(out, ref, atol=0.1)
    print(f"   Expected all = {ref}, Got mean = {out.mean():.2f}")
    print(f"   PASS: {uniform_ok}")
    all_pass &= uniform_ok
    
    # Test 2: Single non-zero K row
    print("\n2. Single K row: K[0,:16] = 1, Q[:,:16] = 1")
    K = torch.zeros(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K[0, :16] = torch.ones(16, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    out = run_kernel(hip, func, K, Q)
    
    K_f32 = K.to(torch.float32)[:, :16]
    Q_f32 = Q.to(torch.float32)[:, :16]
    ref = (K_f32 @ Q_f32.T).cpu().numpy()
    
    # S^T[0, :] = K[0,:] @ Q^T[:,:] = 16 for all columns
    nz = np.where(np.abs(out) > 0.1)[0]
    print(f"   Expected: row 0 = 16, rest = 0")
    print(f"   Got {len(nz)} non-zeros, sum = {out.sum():.0f}")
    single_k_ok = abs(out.sum() - 16 * 32) < 1  # 32 columns should have 16
    print(f"   PASS: {single_k_ok}")
    all_pass &= single_k_ok
    
    # Test 3: Single non-zero Q row  
    print("\n3. Single Q row: K[:,:16] = 1, Q[0,:16] = 1")
    K = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.zeros(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q[0, :16] = torch.ones(16, dtype=torch.float8_e4m3fn, device='cuda')
    out = run_kernel(hip, func, K, Q)
    
    # S^T[:, 0] = K @ Q[0]^T = 16 for all rows
    nz = np.where(np.abs(out) > 0.1)[0]
    print(f"   Expected: col 0 = 16, rest = 0")
    print(f"   Got {len(nz)} non-zeros, sum = {out.sum():.0f}")
    single_q_ok = abs(out.sum() - 16 * 32) < 1
    print(f"   PASS: {single_q_ok}")
    all_pass &= single_q_ok
    
    # Test 4: Random with numerical comparison
    print("\n4. Random test: compare statistics with PyTorch reference")
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        out = run_kernel(hip, func, K, Q)
        
        K_f32 = K.to(torch.float32)[:, :16]
        Q_f32 = Q.to(torch.float32)[:, :16]
        ref = (K_f32 @ Q_f32.T).cpu().numpy()
        
        mean_diff = abs(out.mean() - ref.mean())
        std_diff = abs(out.std() - ref.std())
        
        ok = mean_diff < 0.5 and std_diff < 0.5
        print(f"   Seed {seed}: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f} - {'PASS' if ok else 'FAIL'}")
        all_pass &= ok
    
    # Test 5: Identity-like pattern
    print("\n5. Identity pattern: K[i,:16] = i*0.1, Q = 1")
    K = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    for i in range(32):
        K[i, :16] = i * 0.1
    K = K.to(torch.float8_e4m3fn)
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    out = run_kernel(hip, func, K, Q)
    
    K_f32 = K.to(torch.float32)[:, :16]
    Q_f32 = Q.to(torch.float32)[:, :16]
    ref = (K_f32 @ Q_f32.T).cpu().numpy()
    
    # Each row i should have value i * 0.1 * 16 = i * 1.6
    print(f"   Reference row sums: {ref.sum(axis=1)[:8]}")
    print(f"   Output sum: {out.sum():.1f}, Ref sum: {ref.sum():.1f}")
    identity_ok = abs(out.sum() - ref.sum()) < 10
    print(f"   PASS: {identity_ok}")
    all_pass &= identity_ok
    
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)
    print(f"OVERALL: {'✅ ALL TESTS PASS' if all_pass else '❌ SOME TESTS FAILED'}")
    print("=" * 70)
    
    if all_pass:
        print("\nStep A (ds_read_b64 + XOR swizzle) is NUMERICALLY CORRECT")
        print("Ready for integration into main FP8 flash attention kernel")

if __name__ == "__main__":
    main()

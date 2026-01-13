#!/usr/bin/env python3
"""Step A: Full QK kernel test with ds_read_b64 + XOR swizzle"""

import torch
import subprocess
import ctypes
import numpy as np
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'stepA_full_qk.s', '-o', 'stepA_full_qk.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'stepA_full_qk.co', 'stepA_full_qk.o'], cwd=cwd)
    return str(cwd / 'stepA_full_qk.co')

def main():
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12stepA_full_qkE")
    
    print("Step A: Full QK Kernel (ds_read_b64 + XOR swizzle)")
    print("=" * 70)
    print("Computing S^T = K @ Q^T where K, Q are 32×128 (using first 16 K cols)")
    print()
    
    # Test 1: Uniform K=1, Q=1
    print("Test 1: K = Q = all 1.0")
    K = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    out_cpu = out.cpu().numpy()
    
    # Expected: S^T = K @ Q^T, with K=Q=1, S^T[i,j] = sum_k(1*1) = 16
    print(f"  Expected: 16.0 (sum of 16 ones)")
    print(f"  Actual (lane 0, first 4): {out_cpu[0]:.2f}, {out_cpu[1]:.2f}, {out_cpu[2]:.2f}, {out_cpu[3]:.2f}")
    
    uniform_pass = all(abs(out_cpu[i] - 16.0) < 0.1 for i in range(len(out_cpu)))
    print(f"  Uniform test: {'PASS' if uniform_pass else 'FAIL'}")
    
    # Test 2: Random K and Q
    print()
    print("Test 2: K, Q = random values")
    torch.manual_seed(42)
    K_rand = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q_rand = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out.zero_()
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K_rand.data_ptr()), ctypes.c_void_p(Q_rand.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    out_cpu = out.cpu().numpy()
    
    # Compute reference: S^T = K[:, :16] @ Q[:, :16]^T
    K_f32 = K_rand.to(torch.float32)[:, :16]  # 32×16
    Q_f32 = Q_rand.to(torch.float32)[:, :16]  # 32×16
    ref = (K_f32 @ Q_f32.T).cpu().numpy()     # 32×32
    
    print(f"  Reference S^T[0,0:4]: {ref[0,0]:.2f}, {ref[0,1]:.2f}, {ref[0,2]:.2f}, {ref[0,3]:.2f}")
    print(f"  Actual (lane 0, first 4): {out_cpu[0]:.2f}, {out_cpu[1]:.2f}, {out_cpu[2]:.2f}, {out_cpu[3]:.2f}")
    
    out_mean = out_cpu.mean()
    ref_mean = ref.mean()
    print(f"  Output mean: {out_mean:.2f}, Reference mean: {ref_mean:.2f}")
    
    # More detailed comparison
    out_std = out_cpu.std()
    ref_std = ref.std()
    print(f"  Output std: {out_std:.2f}, Reference std: {ref_std:.2f}")
    
    random_pass = abs(out_mean - ref_mean) < 1.0 and abs(out_std - ref_std) < 1.0
    print(f"  Random test: {'PASS' if random_pass else 'FAIL'}")
    
    hip.hipModuleUnload(module)
    
    if uniform_pass and random_pass:
        print()
        print("=" * 70)
        print("SUCCESS: Full QK baseline is working!")
        print("Ready for Step B: TR_B8 with proper layout")

if __name__ == "__main__":
    main()

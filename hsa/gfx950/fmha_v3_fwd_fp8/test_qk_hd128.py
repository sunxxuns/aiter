#!/usr/bin/env python3
"""Test QK MFMA for HD=128: S[32×32] = Q[32×128] @ K^T[128×32]"""

import torch
import subprocess
import ctypes
import numpy as np

print("Building test_qk_hd128...")
result = subprocess.run([
    '/opt/rocm/llvm/bin/clang++', '-x', 'assembler',
    '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mwavefrontsize64',
    '-c', 'test_qk_hd128.s', '-o', 'test_qk_hd128.o'
], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Compile error:\n{result.stderr}")
    exit(1)

result = subprocess.run([
    '/opt/rocm/llvm/bin/clang++',
    '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mwavefrontsize64',
    'test_qk_hd128.o', '-o', 'test_qk_hd128.co'
], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Link error:\n{result.stderr}")
    exit(1)

hip = ctypes.CDLL('libamdhip64.so')
mod = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(mod), b'test_qk_hd128.co')
func = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func), mod, b'_ZN5aiter12test_qk_hd128E')

print("Kernel loaded\n")

SEQ = 32
HD = 128

print("=" * 70)
print(f"QK MFMA HD={HD} TEST: S[{SEQ}×{SEQ}] = Q[{SEQ}×{HD}] @ K^T[{HD}×{SEQ}]")
print("=" * 70)

def run_test(seed):
    torch.manual_seed(seed)
    
    Q_f = torch.randn(SEQ, HD, dtype=torch.float32) * 0.3
    K_f = torch.randn(SEQ, HD, dtype=torch.float32) * 0.3
    
    Q_fp8 = Q_f.to(torch.float8_e4m3fn)
    K_fp8 = K_f.to(torch.float8_e4m3fn)
    
    Q_gpu = Q_fp8.view(torch.uint8).cuda()
    K_gpu = K_fp8.view(torch.uint8).cuda()
    S = torch.zeros(SEQ, SEQ, dtype=torch.float32, device='cuda')
    
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(S.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(Q_gpu.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(K_gpu.data_ptr())), ctypes.c_void_p),
    )
    
    result = hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args, None)
    if result != 0:
        return None, None, f"Launch failed: {result}"
    
    hip.hipDeviceSynchronize()
    
    # Reference
    S_ref = torch.matmul(Q_fp8.float(), K_fp8.float().T)
    
    return S.cpu(), S_ref, None

# Test 1: Basic execution
print("\n[TEST 1] Basic execution")
S_gpu, S_ref, err = run_test(42)
if err:
    print(f"  ERROR: {err}")
else:
    print(f"  S_ref[0,:8]: {S_ref[0,:8].numpy()}")
    print(f"  S_gpu[0,:8]: {S_gpu[0,:8].numpy()}")
    max_err = (S_gpu - S_ref).abs().max().item()
    print(f"  Max error: {max_err:.6f}")

# Test 2: Multiple seeds
print("\n[TEST 2] Multiple seeds")
for seed in range(5):
    S_gpu, S_ref, err = run_test(seed)
    if err:
        print(f"  Seed {seed}: {err}")
        continue
    max_err = (S_gpu - S_ref).abs().max().item()
    has_nan = torch.isnan(S_gpu).any().item()
    status = "PASS" if max_err < 0.01 and not has_nan else "FAIL"
    print(f"  Seed {seed}: max_err={max_err:.6f} [{status}]")

# Test 3: Check specific elements
print("\n[TEST 3] Specific elements")
S_gpu, S_ref, _ = run_test(42)
for pos in [(0, 0), (0, 31), (15, 15), (31, 0), (31, 31)]:
    r, c = pos
    gpu_val = S_gpu[r, c].item()
    ref_val = S_ref[r, c].item()
    err = abs(gpu_val - ref_val)
    print(f"  S[{r},{c}]: gpu={gpu_val:.4f}, ref={ref_val:.4f}, err={err:.6f}")

print("\n" + "=" * 70)
hip.hipModuleUnload(mod)

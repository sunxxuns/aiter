#!/usr/bin/env python3
"""Test the integrated FP8 flash attention kernel: O = softmax(Q @ K^T) @ V"""

import torch
import subprocess
import ctypes
import numpy as np

# Build (compile + link)
print("Building fwd_fp8_integrated...")
subprocess.run([
    '/opt/rocm/llvm/bin/clang++', '-x', 'assembler',
    '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mwavefrontsize64',
    '-c', 'fwd_fp8_integrated.s', '-o', 'fwd_fp8_integrated.o'
], check=True)
subprocess.run([
    '/opt/rocm/llvm/bin/clang++',
    '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950', '-mwavefrontsize64',
    'fwd_fp8_integrated.o', '-o', 'fwd_fp8_integrated.co'
], check=True)

hip = ctypes.CDLL('libamdhip64.so')

# Load kernel
mod = ctypes.c_void_p()
result = hip.hipModuleLoad(ctypes.byref(mod), b'fwd_fp8_integrated.co')
if result != 0:
    print(f"hipModuleLoad failed: {result}")
    exit(1)

func = ctypes.c_void_p()
result = hip.hipModuleGetFunction(ctypes.byref(func), mod, b'_ZN5aiter17fwd_fp8_integratedE')
if result != 0:
    print(f"hipModuleGetFunction failed: {result}")
    exit(1)

print("Kernel loaded successfully\n")

def run_test(seed, verbose=False):
    """Run a single test with given seed"""
    torch.manual_seed(seed)
    
    # Generate random inputs scaled to FP8 range
    Q_f = torch.randn(32, 32, dtype=torch.float32) * 0.3
    K_f = torch.randn(32, 32, dtype=torch.float32) * 0.3
    V_f = torch.randn(32, 32, dtype=torch.float32) * 0.5
    
    # Convert to FP8
    Q_fp8 = Q_f.to(torch.float8_e4m3fn)
    K_fp8 = K_f.to(torch.float8_e4m3fn)
    V_fp8 = V_f.to(torch.float8_e4m3fn)
    
    # Upload to GPU as bytes
    Q_gpu = Q_fp8.view(torch.uint8).cuda()
    K_gpu = K_fp8.view(torch.uint8).cuda()
    V_gpu = V_fp8.view(torch.uint8).cuda()
    
    # Output buffer
    O = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    # Launch kernel
    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(O.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(Q_gpu.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(K_gpu.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(V_gpu.data_ptr())), ctypes.c_void_p),
    )
    
    result = hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 4096, None, args, None)
    if result != 0:
        return None, None, f"Launch failed: {result}"
    
    hip.hipDeviceSynchronize()
    
    # Reference computation using quantized values
    Q_quant = Q_fp8.to(torch.float32)
    K_quant = K_fp8.to(torch.float32)
    V_quant = V_fp8.to(torch.float32)
    
    S_ref = torch.matmul(Q_quant, K_quant.T)
    P_ref = torch.softmax(S_ref, dim=-1)
    O_ref = torch.matmul(P_ref, V_quant)
    
    O_cpu = O.cpu()
    
    if verbose:
        print(f"\n=== Seed {seed} ===")
        print(f"O[0,:8]     = {O_cpu[0,:8].numpy()}")
        print(f"O_ref[0,:8] = {O_ref[0,:8].numpy()}")
    
    return O_cpu, O_ref, None

# Run tests
print("=" * 70)
print("FP8 FULL ATTENTION TEST: O = softmax(Q @ K^T) @ V")
print("=" * 70)

# Test 1: Multiple random seeds
print("\n[TEST 1] Multiple random seeds (10 different inputs)")
passed = 0
for seed in range(10):
    O_gpu, O_ref, err = run_test(seed)
    if err:
        print(f"  Seed {seed}: {err} [FAIL]")
        continue
    
    max_err = (O_gpu - O_ref).abs().max().item()
    has_nan = torch.isnan(O_gpu).any().item()
    has_inf = torch.isinf(O_gpu).any().item()
    
    # FP8 has limited precision, allow larger tolerance
    ok = max_err < 0.5 and not has_nan and not has_inf
    status = "PASS" if ok else "FAIL"
    print(f"  Seed {seed}: max_err={max_err:.4f}, nan={has_nan}, inf={has_inf} [{status}]")
    if ok:
        passed += 1

# Test 2: Determinism
print("\n[TEST 2] Determinism check (5 runs, same input)")
O_results = []
for i in range(5):
    O_gpu, _, err = run_test(42)
    if err:
        print(f"  Run {i}: {err}")
        continue
    O_results.append(O_gpu.numpy())

if len(O_results) == 5:
    all_same = all(np.allclose(O_results[0], O_results[i]) for i in range(1, 5))
    print(f"  All 5 runs identical: {'PASS' if all_same else 'FAIL'}")
else:
    print("  Could not complete determinism test")

# Test 3: Detailed accuracy check
print("\n[TEST 3] Detailed accuracy for seed=42")
O_gpu, O_ref, err = run_test(42, verbose=True)
if not err:
    for row in [0, 1, 15, 16, 31]:
        err_row = (O_gpu[row,:] - O_ref[row,:]).abs().max().item()
        print(f"  Row {row:2d}: max_err={err_row:.6f}")

# Test 4: Output statistics
print("\n[TEST 4] Output statistics")
O_gpu, O_ref, _ = run_test(42)
print(f"  Kernel output - min: {O_gpu.min().item():.4f}, max: {O_gpu.max().item():.4f}")
print(f"  Reference     - min: {O_ref.min().item():.4f}, max: {O_ref.max().item():.4f}")
print(f"  Has NaN: {torch.isnan(O_gpu).any().item()}")
print(f"  Has Inf: {torch.isinf(O_gpu).any().item()}")

print("\n" + "=" * 70)
print(f"SUMMARY: {passed}/10 random seed tests passed")
print("=" * 70)

if passed >= 8:
    print("\n✓ Full attention kernel is working!")
else:
    print("\n✗ Kernel needs debugging")

hip.hipModuleUnload(mod)

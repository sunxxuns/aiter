#!/usr/bin/env python3
"""Test FP8 QK MFMA with BF16-style swizzle layout."""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Build kernel
print("Building fwd_fp8_swizzle.s...")
os.system("cd /sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8 && "
          "clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 "
          "-c fwd_fp8_swizzle.s -o fwd_fp8_swizzle.o 2>&1 && "
          "ld.lld -shared -o fwd_fp8_swizzle.co fwd_fp8_swizzle.o 2>&1")

# Load kernel
module = ctypes.c_void_p()
res = hip.hipModuleLoad(ctypes.byref(module), b'/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_swizzle.co')
if res != 0:
    print(f"Module load failed: {res}")
    exit(1)

func = ctypes.c_void_p()
res = hip.hipModuleGetFunction(ctypes.byref(func), module, b'_ZN5aiter15fwd_fp8_swizzleE')
if res != 0:
    print(f"Function lookup failed: {res}")
    exit(1)

print("Kernel loaded successfully")

# Test 1: Uniform input
print("\n=== Test 1: Uniform Q=K=1 ===")
S = torch.zeros(64, 16, dtype=torch.float32, device='cuda')  # 64 lanes × 16 values
Q = torch.ones(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
K = torch.ones(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)

args = [
    ctypes.c_void_p(S.data_ptr()),
    ctypes.c_void_p(Q.data_ptr()),
    ctypes.c_void_p(K.data_ptr())
]
args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

# Launch with 256 threads
hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
hip.hipDeviceSynchronize()
err = hip.hipGetLastError()

print(f"Error: {err}")
print(f"P mean: {S.mean().item():.4f} (expected ~0.03125 = 1/32 for uniform)")
print(f"P[0,:8]: {S[0,:8].tolist()}")
print(f"P[32,:8]: {S[32,:8].tolist()}")
print(f"NaN count: {torch.isnan(S).sum().item()}")
print(f"Non-zero: {(S != 0).sum().item()} / {S.numel()}")

# Check if all values are the same (uniform input should give uniform output)
unique_vals = S[~torch.isnan(S)].unique()
print(f"Unique values: {len(unique_vals)}")
if len(unique_vals) <= 5:
    print(f"  Values: {unique_vals.tolist()}")

# Test 2: Check row sums (should be ~1.0 for softmax)
print("\n=== Test 2: Verify row sums ===")
# For softmax, each row should sum to 1.0
# But our output is per-lane (16 values), need to aggregate across lanes
# For now, just check that values are in valid probability range [0, 1]
valid_probs = ((S >= 0) & (S <= 1)).sum().item()
print(f"Values in [0,1]: {valid_probs} / {S.numel()}")

# Test 3: Random input
print("\n=== Test 3: Random input ===")
torch.manual_seed(42)
Q = torch.randn(32, 128, device='cuda').to(torch.float8_e4m3fn)
K = torch.randn(32, 128, device='cuda').to(torch.float8_e4m3fn)
S.zero_()

args = [
    ctypes.c_void_p(S.data_ptr()),
    ctypes.c_void_p(Q.data_ptr()),
    ctypes.c_void_p(K.data_ptr())
]
args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
hip.hipDeviceSynchronize()

print(f"P mean: {S.mean().item():.4f}")
print(f"P range: [{S.min().item():.4f}, {S.max().item():.4f}]")
print(f"NaN count: {torch.isnan(S).sum().item()}")
valid_probs = ((S >= 0) & (S <= 1)).sum().item()
print(f"Values in [0,1]: {valid_probs} / {S.numel()}")

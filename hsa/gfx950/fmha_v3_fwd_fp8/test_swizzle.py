#!/usr/bin/env python3
"""Test FP8 full attention: O = softmax(Q @ K^T) @ V"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

# Build kernel
print("Building fwd_fp8_swizzle.s...")
ret = os.system("cd /sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8 && "
          "clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 "
          "-c fwd_fp8_swizzle.s -o fwd_fp8_swizzle.o 2>&1 && "
          "ld.lld -shared -o fwd_fp8_swizzle.co fwd_fp8_swizzle.o 2>&1")
if ret != 0:
    print("Build failed!")
    exit(1)

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

# Test 1: Uniform input (Q=K=V=1)
print("\n=== Test 1: Uniform Q=K=V=1 ===")
O = torch.zeros(64, 16, dtype=torch.float32, device='cuda')  # 64 lanes × 16 values
Q = torch.ones(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
K = torch.ones(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
V = torch.ones(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)

args = [
    ctypes.c_void_p(O.data_ptr()),
    ctypes.c_void_p(Q.data_ptr()),
    ctypes.c_void_p(K.data_ptr()),
    ctypes.c_void_p(V.data_ptr())
]
args_ptrs = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 20480, None, args_ptrs, None)
hip.hipDeviceSynchronize()
err = hip.hipGetLastError()

print(f"Error: {err}")
print(f"O mean: {O.mean().item():.4f} (expected ~1.0 for uniform V)")
print(f"O[0,:8]: {O[0,:8].tolist()}")
print(f"O[32,:8]: {O[32,:8].tolist()}")
print(f"NaN count: {torch.isnan(O).sum().item()}")
print(f"Non-zero: {(O != 0).sum().item()} / {O.numel()}")

# Test 2: Random input
print("\n=== Test 2: Random input ===")
torch.manual_seed(42)
Q = torch.randn(32, 128, device='cuda').to(torch.float8_e4m3fn)
K = torch.randn(32, 128, device='cuda').to(torch.float8_e4m3fn)
V = torch.randn(32, 128, device='cuda').to(torch.float8_e4m3fn)
O.zero_()

args = [
    ctypes.c_void_p(O.data_ptr()),
    ctypes.c_void_p(Q.data_ptr()),
    ctypes.c_void_p(K.data_ptr()),
    ctypes.c_void_p(V.data_ptr())
]
args_ptrs = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 20480, None, args_ptrs, None)
hip.hipDeviceSynchronize()

print(f"O mean: {O.mean().item():.4f}")
print(f"O range: [{O.min().item():.4f}, {O.max().item():.4f}]")
print(f"NaN count: {torch.isnan(O).sum().item()}")

# Reference computation (simplified)
print("\n=== Reference (softmax @ V) ===")
Q_f32 = Q.to(torch.float32)
K_f32 = K.to(torch.float32)
V_f32 = V.to(torch.float32)
scale = 1.0 / (128 ** 0.5)
S = Q_f32 @ K_f32.T * scale
P = torch.softmax(S, dim=-1)
O_ref = P @ V_f32  # [32, 128]
print(f"Ref O mean: {O_ref[:,:32].mean().item():.4f}")  # Only first 32 cols
print(f"Ref O range: [{O_ref[:,:32].min().item():.4f}, {O_ref[:,:32].max().item():.4f}]")

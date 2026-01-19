#!/usr/bin/env python3
"""Test: Verify TR8 interleaved layout produces correct results.

Layout: Q[row, k] at LDS[(row % 8) + k * 8]
Test with Q[8×128] where Q[r,k] = r*128 + k (sequential values)

After TR8 read at base=row, we should get Q[row, 0:7]:
  = [row*128+0, row*128+1, ..., row*128+7]
"""
import os
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')

import torch
import ctypes
import subprocess
import numpy as np

# Build
base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/low_priority"
print("Building test_tr8_interleaved.s...")
result = subprocess.run([
    '/opt/rocm/llvm/bin/clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
    '-mcpu=gfx950', '-o', 'test_tr8_interleaved.co', 'test_tr8_interleaved.s'
], capture_output=True, cwd=base_dir)

if result.returncode != 0:
    print("Build failed:")
    print(result.stderr.decode())
    exit(1)
print("Build OK")

# Initialize HIP
_ = torch.zeros(1, device='cuda')
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')

module = ctypes.c_void_p()
r = hip.hipModuleLoad(ctypes.byref(module), f"{base_dir}/test_tr8_interleaved.co".encode())
if r != 0:
    print(f"hipModuleLoad failed: {r}")
    exit(1)

func = ctypes.c_void_p()
r = hip.hipModuleGetFunction(ctypes.byref(func), module, b'test_tr8_interleaved')
if r != 0:
    print(f"hipModuleGetFunction failed: {r}")
    exit(1)

# Create input: Q[8×128] with Q[r,k] = r*128 + k (mod 256 for FP8 range)
rows, cols = 8, 128
q_data = np.zeros((rows, cols), dtype=np.uint8)
for r in range(rows):
    for k in range(cols):
        q_data[r, k] = (r * cols + k) % 256

print(f"\nInput Q[{rows}×{cols}]:")
print(f"  Q[0, 0:8] = {list(q_data[0, 0:8])}")
print(f"  Q[1, 0:8] = {list(q_data[1, 0:8])}")
print(f"  Q[7, 0:8] = {list(q_data[7, 0:8])}")

q_input = torch.from_numpy(q_data.flatten()).to(device='cuda')

# Output: 64 threads × 8 bytes = 512 bytes
output = torch.zeros(512, dtype=torch.uint8, device='cuda')

args = [
    ctypes.c_void_p(output.data_ptr()),
    ctypes.c_void_p(q_input.data_ptr()),
]
kargs = (ctypes.c_void_p * len(args))(
    *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
)

# Launch
r = hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 0, None, kargs, None)
if r != 0:
    print(f"Launch failed: {r}")
torch.cuda.synchronize()

# Analyze results
out = output.cpu().numpy()

print("\nTR8 Read Results:")
print("="*60)

# Each thread tid reads Q[tid%8, 0:7]
# So threads 0,8,16,... read row 0
# Threads 1,9,17,... read row 1, etc.

all_correct = True
for tid in range(8):  # First 8 threads, each reads different row
    row = tid % 8
    expected = [q_data[row, k] for k in range(8)]
    actual = list(out[tid*8 : tid*8+8])
    match = expected == actual
    if not match:
        all_correct = False
    status = "✓" if match else "✗"
    print(f"  Thread {tid} (row={row}): expected={expected}, got={actual} {status}")

print()
if all_correct:
    print("SUCCESS: TR8 interleaved layout works correctly!")
else:
    print("FAILURE: TR8 layout mismatch detected")

# Additional verification for other threads
print("\nChecking threads 8-15 (should read same rows as 0-7):")
for tid in range(8, 16):
    row = tid % 8
    expected = [q_data[row, k] for k in range(8)]
    actual = list(out[tid*8 : tid*8+8])
    match = expected == actual
    status = "✓" if match else "✗"
    print(f"  Thread {tid} (row={row}): expected={expected}, got={actual} {status}")

hip.hipModuleUnload(module)

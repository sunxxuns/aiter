#!/usr/bin/env python3
"""Test 128Q QK-only"""

import torch
import subprocess
import ctypes
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'test_128q_qk_only.s', '-o', 'test_128q_qk_only.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'test_128q_qk_only.co', 'test_128q_qk_only.o'], cwd=cwd)
    return str(cwd / 'test_128q_qk_only.co')

def main():
    print("128Q QK-only test")
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter16test_128q_qkonlyE")
    
    # Test: Q=1, K=1 → S = 128 (sum of 128 ones)
    Q = torch.ones(128, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    O = torch.zeros(128, 32, dtype=torch.float32, device='cuda')  # 128 Q rows × 32 K rows
    
    args = [ctypes.c_void_p(O.data_ptr()), ctypes.c_void_p(Q.data_ptr()), ctypes.c_void_p(K.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print("Launching...")
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    print(f"O mean: {O.mean().item():.2f} (expected: 128.0)")
    print(f"O has NaN: {torch.isnan(O).any().item()}")
    
    for wave in range(4):
        wave_mean = O[wave*32:(wave+1)*32, :].mean().item()
        print(f"Wave {wave}: mean={wave_mean:.2f}")
    
    print(f"\nO[0,0:4]: {O[0,0:4].tolist()}")
    print(f"O[32,0:4]: {O[32,0:4].tolist()}")
    print(f"O[64,0:4]: {O[64,0:4].tolist()}")
    print(f"O[96,0:4]: {O[96,0:4].tolist()}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

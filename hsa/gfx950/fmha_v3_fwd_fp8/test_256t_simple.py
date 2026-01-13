#!/usr/bin/env python3
"""Simple test for 256T kernel - just Q loading"""

import torch
import subprocess
import ctypes
from pathlib import Path

def build(name):
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

def main():
    print("Testing 256T kernel compilation...")
    
    co = build("fwd_fp8_256t")
    if not co:
        return
    
    print("Compilation successful!")
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    result = hip.hipModuleLoad(ctypes.byref(module), co.encode())
    print(f"Module load result: {result}")
    
    func = ctypes.c_void_p()
    result = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12fwd_fp8_256tE")
    print(f"Function get result: {result}")
    
    # Small test
    seq_len = 32  # Just 1 tile
    O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda')
    V = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    print(f"Data shapes: O={O.shape}, Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Data pointers: O={O.data_ptr():x}, Q={Q.data_ptr():x}, K={K.data_ptr():x}, V={V.data_ptr():x}")
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_uint32(seq_len)
    ]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print("Launching kernel...")
    result = hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
    print(f"Launch result: {result}")
    
    result = hip.hipDeviceSynchronize()
    print(f"Sync result: {result}")
    
    print(f"O mean: {O.mean().item():.4f}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

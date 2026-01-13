#!/usr/bin/env python3
"""Minimal 256T test - just cooperative Q load"""

import torch
import subprocess
import ctypes
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'test_256t_load.s', '-o', 'test_256t_load.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'test_256t_load.co', 'test_256t_load.o'], cwd=cwd)
    return str(cwd / 'test_256t_load.co')

def main():
    print("Minimal 256T cooperative load test")
    print("=" * 50)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter14test_256t_loadE")
    
    # Q is 256*16 = 4096 bytes = 1024 floats
    Q = torch.arange(1024, dtype=torch.float32, device='cuda')
    out = torch.zeros(1024, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print("Launching kernel with 256 threads...")
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Wave 0 (64 threads) should output first 64*4 = 256 floats
    print(f"Output first 16: {out[:16].tolist()}")
    print(f"Expected first 16: {Q[:16].tolist()}")
    
    match = torch.allclose(out[:256], Q[:256])
    print(f"Match: {match}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

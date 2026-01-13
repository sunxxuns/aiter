#!/usr/bin/env python3
"""Test Q load for 128Q kernel"""

import torch
import subprocess
import ctypes
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'test_128q_qload.s', '-o', 'test_128q_qload.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'test_128q_qload.co', 'test_128q_qload.o'], cwd=cwd)
    return str(cwd / 'test_128q_qload.co')

def main():
    print("128Q Q-load test")
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter16test_128q_qloadE")
    
    O = torch.zeros(128, 128, dtype=torch.float32, device='cuda')
    Q = torch.arange(128*128, dtype=torch.float32, device='cuda').view(128, 128).to(torch.float8_e4m3fn)
    
    args = [ctypes.c_void_p(O.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print("Launching...")
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    print(f"O shape: {O.shape}")
    
    # Each wave should output its wave_id
    for wave in range(4):
        wave_vals = O[wave*32:(wave+1)*32, :4]
        print(f"Wave {wave}: O[{wave*32}, 0:4] = {wave_vals[0].tolist()}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

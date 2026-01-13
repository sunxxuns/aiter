#!/usr/bin/env python3
"""Simple test - just load Q and output"""

import torch
import subprocess
import ctypes
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'fwd_fp8_128q.s', '-o', 'fwd_fp8_128q.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'fwd_fp8_128q.co', 'fwd_fp8_128q.o'], cwd=cwd)
    return str(cwd / 'fwd_fp8_128q.co')

def main():
    print("Simple 128Q test")
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    result = hip.hipModuleLoad(ctypes.byref(module), co.encode())
    print(f"Module load: {result}")
    
    func = ctypes.c_void_p()
    result = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13fwd_fp8_128qE")
    print(f"Function get: {result}")
    
    # Minimal test
    seq_len = 32  # Just 1 K-tile
    print(f"\nseq_len={seq_len}")
    
    # Ensure contiguous and proper alignment
    O = torch.zeros(128, 128, dtype=torch.float32, device='cuda').contiguous()
    Q = torch.ones(128, 128, dtype=torch.float8_e4m3fn, device='cuda').contiguous()
    K = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda').contiguous()
    V = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda').contiguous()
    
    print(f"Tensor sizes: O={O.numel()*4}B, Q={Q.numel()}B, K={K.numel()}B, V={V.numel()}B")
    print(f"Pointers: O={O.data_ptr():x}, Q={Q.data_ptr():x}, K={K.data_ptr():x}, V={V.data_ptr():x}")
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_uint32(seq_len)
    ]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print("Launching with 256 threads, 65536 LDS...")
    result = hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    print(f"Launch result: {result}")
    
    result = hip.hipDeviceSynchronize()
    print(f"Sync result: {result}")
    
    print(f"O mean: {O.mean().item():.4f}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

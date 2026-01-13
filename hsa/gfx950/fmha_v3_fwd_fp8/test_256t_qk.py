#!/usr/bin/env python3
"""Test 256T QK-only kernel"""

import torch
import subprocess
import ctypes
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'test_256t_qk.s', '-o', 'test_256t_qk.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'test_256t_qk.co', 'test_256t_qk.o'], cwd=cwd)
    return str(cwd / 'test_256t_qk.co')

def main():
    print("256T QK-only test")
    print("=" * 50)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12test_256t_qkE")
    
    # K[32×128], Q[32×128], out[64×16]
    K = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print("Launching 256T QK kernel...")
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    result = out.cpu().numpy()
    print(f"Output mean: {result.mean():.2f}")
    print(f"Expected: 128.0 (sum of 128 ones)")
    print(f"First 4 values: {result[:4]}")
    
    # Compare with 64T kernel (stepA)
    co_64 = str(Path(__file__).parent / 'stepA_full_qk.co')
    if Path(co_64).exists():
        module_64 = ctypes.c_void_p()
        hip.hipModuleLoad(ctypes.byref(module_64), co_64.encode())
        func_64 = ctypes.c_void_p()
        hip.hipModuleGetFunction(ctypes.byref(func_64), module_64, b"_ZN5aiter12stepA_full_qkE")
        
        out_64 = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
        args_64 = [ctypes.c_void_p(out_64.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
        args_ptrs_64 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_64])
        
        hip.hipModuleLaunchKernel(func_64, 1, 1, 1, 64, 1, 1, 16384, None, args_ptrs_64, None)
        hip.hipDeviceSynchronize()
        
        print(f"\n64T kernel mean: {out_64.cpu().numpy().mean():.2f}")
        hip.hipModuleUnload(module_64)
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

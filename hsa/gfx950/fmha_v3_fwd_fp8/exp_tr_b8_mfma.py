#!/usr/bin/env python3
"""Test TR_B8 with MFMA - K=8 only for simplicity"""

import torch
import subprocess
import ctypes
import numpy as np
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'exp_tr_b8_mfma.s', '-o', 'exp_tr_b8_mfma.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'exp_tr_b8_mfma.co', 'exp_tr_b8_mfma.o'], cwd=cwd)
    return str(cwd / 'exp_tr_b8_mfma.co')

def main():
    print("=" * 70)
    print("TR_B8 + MFMA Test (K=8)")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter14exp_tr_b8_mfmaE")
    
    # K[32×16], Q[32×8] - but we only use K=8 for this test
    K = torch.ones(32, 16, dtype=torch.float8_e4m3fn, device='cuda')
    Q = torch.ones(32, 8, dtype=torch.float8_e4m3fn, device='cuda')
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    result = out.cpu().numpy()
    
    print(f"\nUniform K=Q=1 test:")
    print(f"  Expected: 8.0 (K=8 dot product)")
    print(f"  Actual mean: {result.mean():.2f}")
    print(f"  Actual first 4: {result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}, {result[3]:.2f}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

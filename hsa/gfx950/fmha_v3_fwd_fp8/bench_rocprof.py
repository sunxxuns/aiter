#!/usr/bin/env python3
"""
Run rocprof on existing kernel to measure LDS bank conflicts.
"""

import torch
import ctypes
import os
import sys

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def run_kernel(iterations=100):
    """Run the preload kernel multiple times."""
    
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co"
    func_name = b"_ZN5aiter17fwd_fp8_qk_preloadE"
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    
    seq_len = 64
    
    # Create test data
    Q = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Warmup
    for _ in range(3):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Run iterations for profiling
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    print(f"Ran {iterations} iterations")
    print(f"Output[0]: {O[0].item()}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    run_kernel()

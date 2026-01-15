#!/usr/bin/env python3
"""Test the known-working fwd_fp8_kloop kernel"""
import os
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')

import torch
import ctypes
import struct

hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")

# Set up function signatures
hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
hip.hipModuleLoad.restype = ctypes.c_int

hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
hip.hipModuleGetFunction.restype = ctypes.c_int

hip.hipModuleLaunchKernel.argtypes = [
    ctypes.c_void_p,  # function
    ctypes.c_uint,    # gridDimX
    ctypes.c_uint,    # gridDimY
    ctypes.c_uint,    # gridDimZ
    ctypes.c_uint,    # blockDimX
    ctypes.c_uint,    # blockDimY
    ctypes.c_uint,    # blockDimZ
    ctypes.c_uint,    # sharedMemBytes
    ctypes.c_void_p,  # stream
    ctypes.POINTER(ctypes.c_void_p),  # kernelParams
    ctypes.POINTER(ctypes.c_void_p),  # extra
]
hip.hipModuleLaunchKernel.restype = ctypes.c_int

hip.hipDeviceSynchronize.argtypes = []
hip.hipDeviceSynchronize.restype = ctypes.c_int

hip.hipHostMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
hip.hipHostMalloc.restype = ctypes.c_int

hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
hip.hipModuleUnload.restype = ctypes.c_int

def load_module(path):
    module = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    return module

def get_function(module, name):
    func = ctypes.c_void_p()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return func

def launch_kernel(func, grid, block, kernel_args, shared_mem=0):
    ret = hip.hipModuleLaunchKernel(
        func,
        ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(grid[2]),
        ctypes.c_uint(block[0]), ctypes.c_uint(block[1]), ctypes.c_uint(block[2]),
        ctypes.c_uint(shared_mem),
        None,  # stream
        kernel_args,  # kernelParams
        None   # extra
    )
    if ret != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {ret}")

def main():
    print("Testing fwd_fp8_kloop kernel...", flush=True)
    
    # Kernel params from the .s file:
    # O [32×128] F32, Q [32×128] FP8, K [seq×128] FP8, V [seq×128] FP8, seq_len
    # 64 threads (1 wave), 12288 LDS bytes
    
    seq_len = 64
    hd = 128
    rows = 32
    
    # Create tensors
    print("Creating tensors...", flush=True)
    Q = torch.randn(rows, hd, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(seq_len, hd, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    V = torch.randn(seq_len, hd, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(rows, hd, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    print(f"  Q: {Q.shape}, K: {K.shape}, V: {V.shape}, O: {O.shape}", flush=True)
    
    # Load kernel
    co_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_kloop.co"
    kernel_name = "_ZN5aiter13fwd_fp8_kloopE"
    
    print(f"Loading {co_path}...", flush=True)
    module = load_module(co_path)
    func = get_function(module, kernel_name)
    print("Kernel loaded", flush=True)
    
    # Use kernelParams approach (like bench_kloop.py)
    print("Packing args...", flush=True)
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_uint32(seq_len)
    ]
    kernel_args = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    print(f"  Args ready", flush=True)
    
    # Launch: 1 wave of 64 threads, 12288 LDS
    grid = (1, 1, 1)
    block = (64, 1, 1)
    shared_mem = 12288
    
    print(f"Launching kernel...", flush=True)
    launch_kernel(func, grid, block, kernel_args, shared_mem)
    print("Launch returned", flush=True)
    
    torch.cuda.synchronize()
    print("Sync done", flush=True)
    
    print(f"Output stats: mean={O.mean().item():.4f}, std={O.std().item():.4f}")
    
    hip.hipModuleUnload(module)
    print("Done!")

if __name__ == "__main__":
    main()

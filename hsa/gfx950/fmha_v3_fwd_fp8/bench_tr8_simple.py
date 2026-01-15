#!/usr/bin/env python3
"""Simple benchmark for TR8 kernel variants"""
import os
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')

import torch
import ctypes
import struct
import time

hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")

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

def launch_kernel(func, grid, block, args_ptr, shared_mem=0):
    ret = hip.hipModuleLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_mem, None, args_ptr, None
    )
    if ret != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {ret}")

def test_kernel(kernel_name, co_path, num_k_tiles=8):
    """Test a single TR8 kernel"""
    print(f"\nTesting {kernel_name}...", flush=True)
    
    # Test dimensions
    M, N = 32, 32  # Output dimensions
    K_dim = num_k_tiles * 16  # 128 when num_k_tiles=8
    
    # Create test data
    # Q needs: 4 loads × 256 threads × 16 bytes = 16384 bytes
    # K needs: num_k_tiles loads × 256 threads × 16 bytes = 32768 bytes for num_k_tiles=8
    print(f"  Creating tensors...", flush=True)
    Q_size = 256 * 4 * 16  # 16384 bytes
    K_size = 256 * num_k_tiles * 16  # 32768 bytes for num_k_tiles=8
    O_size = 256 * 4 * 16  # 16384 float32 values = 65536 bytes
    
    Q = torch.randn(Q_size, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(K_size, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(O_size, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    print(f"  Tensors created: Q={Q.numel()} B, K={K.numel()} B, O={O.numel()*4} B", flush=True)
    
    # Load kernel
    if not os.path.exists(co_path):
        print(f"  File not found: {co_path}", flush=True)
        return None
    
    print(f"  Loading module...", flush=True)
    module = load_module(co_path)
    print(f"  Getting function...", flush=True)
    func = get_function(module, kernel_name)
    print(f"  Kernel loaded", flush=True)
    
    # Use kernelParams approach
    print(f"  Packing args...", flush=True)
    print(f"    O ptr: 0x{O.data_ptr():x}", flush=True)
    print(f"    Q ptr: 0x{Q.data_ptr():x}", flush=True)
    print(f"    K ptr: 0x{K.data_ptr():x}", flush=True)
    print(f"    num_k_tiles: {num_k_tiles}", flush=True)
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(num_k_tiles)
    ]
    kernel_args = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    print(f"  Args ready", flush=True)
    
    # Launch config
    grid = (1, 1, 1)
    block = (256, 1, 1)  # 4 waves of 64 threads
    
    # Warmup
    print(f"  Starting warmup...", flush=True)
    try:
        for i in range(5):
            print(f"    Warmup iter {i}...", flush=True)
            launch_kernel(func, grid, block, kernel_args, 65536)
            hip.hipDeviceSynchronize()
            print(f"    Warmup iter {i} done", flush=True)
    except Exception as e:
        print(f"  Warmup failed: {e}", flush=True)
        hip.hipModuleUnload(module)
        return None
    print(f"  Warmup complete", flush=True)
    
    # Benchmark
    iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        launch_kernel(func, grid, block, kernel_args, 65536)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    time_us = (elapsed_ms / iters) * 1000
    
    # Calculate TFLOPS
    flops = 2 * M * N * K_dim * num_k_tiles
    tflops = flops / (time_us * 1e-6) / 1e12
    
    print(f"  Time: {time_us:.2f} us")
    print(f"  TF/s: {tflops:.3f}")
    
    hip.hipModuleUnload(module)
    return time_us

def main():
    print("=== TR8 Kernel Benchmark ===", flush=True)
    
    # First do a simple GPU warmup
    print("GPU warmup...", flush=True)
    x = torch.randn(100, device='cuda')
    y = x * 2
    del x, y
    torch.cuda.synchronize()
    print("GPU ready", flush=True)
    
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    
    # Build kernels first
    print("\nBuilding kernels...", flush=True)
    os.system(f"cd {base_dir} && clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o fwd_fp8_qk_v8swizzle.co fwd_fp8_qk_v8swizzle.s 2>&1")
    os.system(f"cd {base_dir} && clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o fwd_fp8_qk_tr8scaled.co fwd_fp8_qk_tr8scaled.s 2>&1")
    print("Kernels built", flush=True)
    
    kernels = [
        ("fwd_fp8_qk_v8swizzle", f"{base_dir}/fwd_fp8_qk_v8swizzle.co"),
        ("fwd_fp8_qk_tr8scaled", f"{base_dir}/fwd_fp8_qk_tr8scaled.co"),
    ]
    
    for kernel_name, co_path in kernels:
        print(f"\n>>> Testing {kernel_name}...", flush=True)
        try:
            test_kernel(kernel_name, co_path)
        except Exception as e:
            print(f"  Exception: {e}", flush=True)

if __name__ == "__main__":
    main()

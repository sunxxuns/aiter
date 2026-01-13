#!/usr/bin/env python3
"""Benchmark 256T vs 64T QK kernel"""

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

def benchmark(hip, func, args_ptrs, num_blocks, threads, warmup=50, iterations=500):
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start))
    hip.hipEventCreate(ctypes.byref(end))
    
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, threads, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    hip.hipEventRecord(start, None)
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, threads, 1, 1, 32768, None, args_ptrs, None)
    hip.hipEventRecord(end, None)
    hip.hipEventSynchronize(end)
    
    elapsed = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(end)
    
    return elapsed.value * 1000 / iterations

def main():
    print("=" * 70)
    print("Benchmark: 64T vs 256T QK Kernel")
    print("=" * 70)
    
    co_64t = build("stepA_full_qk")
    co_256t = build("test_256t_qk")
    
    if not co_64t or not co_256t:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module_64t = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_64t), co_64t.encode())
    func_64t = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_64t), module_64t, b"_ZN5aiter12stepA_full_qkE")
    
    module_256t = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_256t), co_256t.encode())
    func_256t = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_256t), module_256t, b"_ZN5aiter12test_256t_qkE")
    
    # Data
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Per-kernel FLOPS: 8 MFMAs × 32×32×16 × 2 = 524288
    ops_per_kernel = 8 * 32 * 32 * 16 * 2
    
    print(f"\n{'Blocks':<10} {'64T (µs)':<12} {'256T (µs)':<12} {'64T TF/s':<12} {'256T TF/s':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for num_blocks in [256, 1024, 4096, 16384]:
        t_64 = benchmark(hip, func_64t, args_ptrs, num_blocks, 64)
        t_256 = benchmark(hip, func_256t, args_ptrs, num_blocks, 256)
        
        tflops_64 = (ops_per_kernel * num_blocks) / (t_64 * 1e-6) / 1e12
        tflops_256 = (ops_per_kernel * num_blocks) / (t_256 * 1e-6) / 1e12
        speedup = t_64 / t_256
        
        print(f"{num_blocks:<10} {t_64:<12.2f} {t_256:<12.2f} {tflops_64:<12.1f} {tflops_256:<12.1f} {speedup:<10.2f}x")
    
    hip.hipModuleUnload(module_64t)
    hip.hipModuleUnload(module_256t)
    
    print("\n" + "=" * 70)
    print("Note: 256T has 4x loading bandwidth but redundant MFMA (same compute)")
    print("=" * 70)

if __name__ == "__main__":
    main()

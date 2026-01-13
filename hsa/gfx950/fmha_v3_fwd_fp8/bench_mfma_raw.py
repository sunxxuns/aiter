#!/usr/bin/env python3
"""Measure raw MFMA throughput (no memory)"""

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
    print("=" * 70)
    print("Raw MFMA Throughput Test (no memory operations)")
    print("=" * 70)
    
    co = build("bench_mfma_raw")
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13bench_mfma_rawE")
    
    # Each kernel: 256 MFMAs × 32×32×16 × 2 = 268,435,456 ops
    ops_per_kernel = 256 * 32 * 32 * 16 * 2
    
    print(f"\nOps per kernel: {ops_per_kernel:,}")
    print(f"MFMAs per kernel: 256")
    
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start))
    hip.hipEventCreate(ctypes.byref(end))
    
    # Test at different scales
    print(f"\n{'Blocks':<10} {'Time (µs)':<12} {'TF/s':<12} {'% of 2400 peak':<15}")
    print("-" * 55)
    
    for num_blocks in [64, 256, 1024, 4096, 16384]:
        # Warmup
        for _ in range(50):
            hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 0, None, None, None)
        hip.hipDeviceSynchronize()
        
        # Timed run
        iterations = 200
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 0, None, None, None)
        hip.hipEventRecord(end, None)
        hip.hipEventSynchronize(end)
        
        elapsed = ctypes.c_float()
        hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
        time_us = elapsed.value * 1000 / iterations
        
        total_ops = ops_per_kernel * num_blocks
        tflops = total_ops / (time_us * 1e-6) / 1e12
        pct_peak = tflops / 2400 * 100
        
        print(f"{num_blocks:<10} {time_us:<12.2f} {tflops:<12.1f} {pct_peak:<15.1f}%")
    
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(end)
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)
    print("Note: Peak FP8 MFMA = 2400 TF/s (theoretical)")
    print("=" * 70)

if __name__ == "__main__":
    main()

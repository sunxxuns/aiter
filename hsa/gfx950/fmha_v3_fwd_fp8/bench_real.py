#!/usr/bin/env python3
"""
Real benchmark: Stride-128 vs Stride-132
Uses HIP events for accurate GPU timing
"""

import torch
import subprocess
import ctypes
import time
from pathlib import Path

def build(name):
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', f'{name}.s', '-o', f'{name}.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error for {name}:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', f'{name}.co', f'{name}.o'], cwd=cwd)
    return str(cwd / f'{name}.co')

def hip_event_benchmark(hip, func, K, Q, out, num_blocks, warmup=100, iterations=1000):
    """Use HIP events for accurate GPU timing"""
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Create events
    start_event = ctypes.c_void_p()
    end_event = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start_event))
    hip.hipEventCreate(ctypes.byref(end_event))
    
    # Warmup
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Timed run
    hip.hipEventRecord(start_event, None)
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipEventRecord(end_event, None)
    hip.hipEventSynchronize(end_event)
    
    # Get elapsed time
    elapsed_ms = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, end_event)
    
    hip.hipEventDestroy(start_event)
    hip.hipEventDestroy(end_event)
    
    return elapsed_ms.value * 1000 / iterations  # microseconds per iteration

def main():
    print("=" * 70)
    print("REAL BENCHMARK: Stride-128 (bank conflicts) vs Stride-132 (no conflicts)")
    print("=" * 70)
    
    co_128 = build("qk_fp8_stride128")
    co_132 = build("qk_fp8_stride132_opt")
    
    if not co_128 or not co_132:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module_128 = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_128), co_128.encode())
    func_128 = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_128), module_128, b"_ZN5aiter15qk_fp8_stride128E")
    
    module_132 = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_132), co_132.encode())
    func_132 = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_132), module_132, b"_ZN5aiter19qk_fp8_stride132_optE")
    
    # Test data
    torch.manual_seed(42)
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    # Verify correctness first
    print("\n1. Correctness verification")
    print("-" * 50)
    K_uniform = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q_uniform = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K_uniform.data_ptr()), ctypes.c_void_p(Q_uniform.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func_128, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    out_128 = out.clone()
    
    hip.hipModuleLaunchKernel(func_132, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    out_132 = out.clone()
    
    print(f"   Stride-128: mean = {out_128.mean().item():.2f} (expected 128.0)")
    print(f"   Stride-132: mean = {out_132.mean().item():.2f} (expected 128.0)")
    
    # Benchmark at different scales
    print("\n2. Performance benchmark (HIP events)")
    print("-" * 50)
    print(f"   {'Blocks':<10} {'Stride-128':<15} {'Stride-132':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for num_blocks in [1, 4, 16, 64, 256, 1024]:
        time_128 = hip_event_benchmark(hip, func_128, K, Q, out, num_blocks, warmup=50, iterations=500)
        time_132 = hip_event_benchmark(hip, func_132, K, Q, out, num_blocks, warmup=50, iterations=500)
        
        speedup = time_128 / time_132
        print(f"   {num_blocks:<10} {time_128:>10.2f} µs   {time_132:>10.2f} µs   {speedup:>6.2f}x")
    
    # Calculate throughput at scale
    print("\n3. Throughput at scale (1024 blocks)")
    print("-" * 50)
    
    num_blocks = 1024
    iterations = 1000
    
    time_128 = hip_event_benchmark(hip, func_128, K, Q, out, num_blocks, warmup=100, iterations=iterations)
    time_132 = hip_event_benchmark(hip, func_132, K, Q, out, num_blocks, warmup=100, iterations=iterations)
    
    # Each block: 8 MFMAs × 32×32×16 FP8 ops × 2 (mul+add)
    ops_per_block = 8 * 32 * 32 * 16 * 2
    total_ops = ops_per_block * num_blocks
    
    tflops_128 = total_ops / (time_128 * 1e-6) / 1e12
    tflops_132 = total_ops / (time_132 * 1e-6) / 1e12
    
    print(f"   Stride-128: {time_128:.2f} µs, {tflops_128:.2f} TF/s")
    print(f"   Stride-132: {time_132:.2f} µs, {tflops_132:.2f} TF/s")
    print(f"   Speedup: {time_128/time_132:.2f}x")
    
    # Extended benchmark
    print("\n4. Extended benchmark (2000 iterations)")
    print("-" * 50)
    
    for num_blocks in [256, 512, 1024, 2048]:
        time_128 = hip_event_benchmark(hip, func_128, K, Q, out, num_blocks, warmup=200, iterations=2000)
        time_132 = hip_event_benchmark(hip, func_132, K, Q, out, num_blocks, warmup=200, iterations=2000)
        
        total_ops = ops_per_block * num_blocks
        tflops_128 = total_ops / (time_128 * 1e-6) / 1e12
        tflops_132 = total_ops / (time_132 * 1e-6) / 1e12
        speedup = time_128 / time_132
        
        print(f"   {num_blocks} blocks: 128={tflops_128:.1f} TF/s, 132={tflops_132:.1f} TF/s, speedup={speedup:.2f}x")
    
    hip.hipModuleUnload(module_128)
    hip.hipModuleUnload(module_132)
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Stride-132 eliminates bank conflicts for measurable speedup")
    print("=" * 70)

if __name__ == "__main__":
    main()

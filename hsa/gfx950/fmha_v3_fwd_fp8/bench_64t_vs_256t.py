#!/usr/bin/env python3
"""Compare 64T (1 wave) vs 256T (4 waves) FP8 QK kernels"""

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
        print(f"Compile error for {name}:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', f'{name}.co', f'{name}.o'], cwd=cwd)
    return str(cwd / f'{name}.co')

def benchmark(hip, func, K, Q, out, num_blocks, threads_per_block, warmup=100, iterations=500):
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start))
    hip.hipEventCreate(ctypes.byref(end))
    
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, threads_per_block, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    hip.hipEventRecord(start, None)
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, threads_per_block, 1, 1, 32768, None, args_ptrs, None)
    hip.hipEventRecord(end, None)
    hip.hipEventSynchronize(end)
    
    elapsed = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(end)
    
    return elapsed.value * 1000 / iterations

def main():
    print("=" * 70)
    print("BENCHMARK: 64T (1 wave) vs 256T (4 waves)")
    print("=" * 70)
    
    co_64t = build("qk_fp8_stride128")
    co_256t = build("qk_fp8_256t")
    
    if not co_64t or not co_256t:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module_64t = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_64t), co_64t.encode())
    func_64t = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_64t), module_64t, b"_ZN5aiter15qk_fp8_stride128E")
    
    module_256t = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_256t), co_256t.encode())
    func_256t = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_256t), module_256t, b"_ZN5aiter11qk_fp8_256tE")
    
    torch.manual_seed(42)
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    # Correctness
    print("\n1. Correctness (K=Q=1)")
    print("-" * 50)
    K_u = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q_u = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K_u.data_ptr()), ctypes.c_void_p(Q_u.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func_64t, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    print(f"   64T:  mean = {out.mean().item():.2f} (expected 128.0)")
    
    out.zero_()
    hip.hipModuleLaunchKernel(func_256t, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    print(f"   256T: mean = {out.mean().item():.2f} (expected 128.0)")
    
    # Benchmark
    print("\n2. Performance")
    print("-" * 50)
    print(f"   {'Blocks':<10} {'64T (1 wave)':<15} {'256T (4 waves)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    ops_per_block = 8 * 32 * 32 * 16 * 2
    
    for num_blocks in [64, 256, 512, 1024, 2048, 4096]:
        t_64t = benchmark(hip, func_64t, K, Q, out, num_blocks, 64)
        t_256t = benchmark(hip, func_256t, K, Q, out, num_blocks, 256)
        speedup = t_64t / t_256t
        print(f"   {num_blocks:<10} {t_64t:>10.2f} µs   {t_256t:>10.2f} µs   {speedup:>6.2f}x")
    
    # TF/s at scale
    print("\n3. Throughput at 4096 blocks")
    print("-" * 50)
    
    t_64t = benchmark(hip, func_64t, K, Q, out, 4096, 64, warmup=200, iterations=1000)
    t_256t = benchmark(hip, func_256t, K, Q, out, 4096, 256, warmup=200, iterations=1000)
    
    total_ops = ops_per_block * 4096
    tflops_64t = total_ops / (t_64t * 1e-6) / 1e12
    tflops_256t = total_ops / (t_256t * 1e-6) / 1e12
    
    print(f"   64T:  {t_64t:.2f} µs, {tflops_64t:.1f} TF/s")
    print(f"   256T: {t_256t:.2f} µs, {tflops_256t:.1f} TF/s")
    print(f"   Speedup: {t_64t/t_256t:.2f}x")
    
    hip.hipModuleUnload(module_64t)
    hip.hipModuleUnload(module_256t)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

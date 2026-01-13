#!/usr/bin/env python3
"""Test and benchmark optimized stride-132 kernel"""

import torch
import subprocess
import ctypes
import time
import numpy as np
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

def run_kernel(hip, func, K, Q, out):
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()

def benchmark(hip, func, K, Q, out, warmup=50, iterations=500):
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    end = time.perf_counter()
    
    return (end - start) * 1e6 / iterations

def main():
    print("Stride-132 Kernel: Test and Benchmark")
    print("=" * 70)
    
    co_orig = build("qk_fp8_stride132")
    co_opt = build("qk_fp8_stride132_opt")
    
    if not co_orig or not co_opt:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module_orig = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_orig), co_orig.encode())
    func_orig = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_orig), module_orig, b"_ZN5aiter15qk_fp8_stride132E")
    
    module_opt = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_opt), co_opt.encode())
    func_opt = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_opt), module_opt, b"_ZN5aiter19qk_fp8_stride132_optE")
    
    # Test data
    torch.manual_seed(42)
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out_orig = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    out_opt = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    # Correctness test
    print("\n1. Correctness Test (Uniform K=Q=1)")
    print("-" * 50)
    K_uniform = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q_uniform = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    run_kernel(hip, func_orig, K_uniform, Q_uniform, out_orig)
    run_kernel(hip, func_opt, K_uniform, Q_uniform, out_opt)
    
    print(f"  Original: mean={out_orig.mean().item():.2f}")
    print(f"  Optimized: mean={out_opt.mean().item():.2f}")
    print(f"  Expected: 128.0")
    print(f"  Match: {abs(out_orig.mean().item() - 128.0) < 0.1 and abs(out_opt.mean().item() - 128.0) < 0.1}")
    
    # Random test
    print("\n2. Correctness Test (Random)")
    print("-" * 50)
    run_kernel(hip, func_orig, K, Q, out_orig)
    run_kernel(hip, func_opt, K, Q, out_opt)
    
    K_f32 = K.to(torch.float32)
    Q_f32 = Q.to(torch.float32)
    ref = (K_f32 @ Q_f32.T).cpu().numpy()
    
    print(f"  Reference mean: {ref.mean():.3f}")
    print(f"  Original mean: {out_orig.cpu().numpy().mean():.3f}")
    print(f"  Optimized mean: {out_opt.cpu().numpy().mean():.3f}")
    
    orig_diff = abs(out_orig.cpu().numpy().mean() - ref.mean())
    opt_diff = abs(out_opt.cpu().numpy().mean() - ref.mean())
    print(f"  Original diff: {orig_diff:.6f}")
    print(f"  Optimized diff: {opt_diff:.6f}")
    
    # Benchmark
    print("\n3. Benchmark")
    print("-" * 50)
    
    time_orig = benchmark(hip, func_orig, K, Q, out_orig)
    time_opt = benchmark(hip, func_opt, K, Q, out_opt)
    
    ops_per_kernel = 8 * 32 * 32 * 16 * 2
    tflops_orig = ops_per_kernel / (time_orig * 1e-6) / 1e12
    tflops_opt = ops_per_kernel / (time_opt * 1e-6) / 1e12
    
    print(f"  Original:  {time_orig:.2f} µs ({tflops_orig:.3f} TF/s)")
    print(f"  Optimized: {time_opt:.2f} µs ({tflops_opt:.3f} TF/s)")
    print(f"  Speedup: {time_orig/time_opt:.2f}x")
    
    hip.hipModuleUnload(module_orig)
    hip.hipModuleUnload(module_opt)

if __name__ == "__main__":
    main()

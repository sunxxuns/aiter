#!/usr/bin/env python3
"""Benchmark stride-132 vs stride-128 for FP8 QK MFMA"""

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

def benchmark(hip, func, K, Q, out, warmup=20, iterations=100):
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Warmup
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    end = time.perf_counter()
    
    return (end - start) * 1e6 / iterations  # microseconds

def main():
    print("Benchmark: Stride-132 (Zero Bank Conflicts)")
    print("=" * 70)
    
    co_132 = build("qk_fp8_stride132")
    co_A = build("stepA_full_qk")  # Original K=16 only kernel
    
    if not co_132:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module_132 = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_132), co_132.encode())
    func_132 = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_132), module_132, b"_ZN5aiter15qk_fp8_stride132E")
    
    # Test data
    torch.manual_seed(42)
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    # Benchmark stride-132
    time_132 = benchmark(hip, func_132, K, Q, out, warmup=50, iterations=500)
    print(f"\nStride-132 (HD=128, 8 MFMAs): {time_132:.2f} µs/kernel")
    
    # Calculate theoretical performance
    # 8 MFMAs × 32×32×16 = 8 × 32768 = 262144 FP8 FMAs
    # Each FMA = 2 ops (multiply + add)
    ops_per_kernel = 8 * 32 * 32 * 16 * 2
    tflops = ops_per_kernel / (time_132 * 1e-6) / 1e12
    print(f"  Compute: {ops_per_kernel:,} ops")
    print(f"  Throughput: {tflops:.2f} TF/s")
    
    # Extended benchmark
    print("\nExtended benchmark (1000 iterations):")
    time_ext = benchmark(hip, func_132, K, Q, out, warmup=100, iterations=1000)
    tflops_ext = ops_per_kernel / (time_ext * 1e-6) / 1e12
    print(f"  Time: {time_ext:.2f} µs/kernel")
    print(f"  Throughput: {tflops_ext:.2f} TF/s")
    
    # Verify correctness
    print("\nVerify correctness:")
    hip.hipModuleLaunchKernel(func_132, 1, 1, 1, 64, 1, 1, 32768, None, 
        (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in 
            [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]]), None)
    hip.hipDeviceSynchronize()
    
    K_f32 = K.to(torch.float32)
    Q_f32 = Q.to(torch.float32)
    ref = (K_f32 @ Q_f32.T).cpu().numpy()
    out_cpu = out.cpu().numpy()
    
    print(f"  Output mean: {out_cpu.mean():.3f}, Reference mean: {ref.mean():.3f}")
    print(f"  Match: {abs(out_cpu.mean() - ref.mean()) < 0.1}")
    
    hip.hipModuleUnload(module_132)

if __name__ == "__main__":
    main()

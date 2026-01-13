#!/usr/bin/env python3
"""
Complete benchmark: Stride-128 vs Stride-132 (mul) vs Stride-132 (fast shifts)
"""

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

def benchmark(hip, func, K, Q, out, num_blocks, warmup=100, iterations=1000):
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(Q.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    start_event = ctypes.c_void_p()
    end_event = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start_event))
    hip.hipEventCreate(ctypes.byref(end_event))
    
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    hip.hipEventRecord(start_event, None)
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
    hip.hipEventRecord(end_event, None)
    hip.hipEventSynchronize(end_event)
    
    elapsed_ms = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, end_event)
    hip.hipEventDestroy(start_event)
    hip.hipEventDestroy(end_event)
    
    return elapsed_ms.value * 1000 / iterations

def main():
    print("=" * 80)
    print("BENCHMARK: LDS Stride Comparison for FP8 QK MFMA")
    print("=" * 80)
    
    kernels = {
        "stride128": ("qk_fp8_stride128", "_ZN5aiter15qk_fp8_stride128E"),
        "stride132_mul": ("qk_fp8_stride132_opt", "_ZN5aiter19qk_fp8_stride132_optE"),
        "stride132_fast": ("qk_fp8_stride132_fast", "_ZN5aiter20qk_fp8_stride132_fastE"),
    }
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    funcs = {}
    modules = {}
    
    for name, (file, symbol) in kernels.items():
        co = build(file)
        if not co:
            return
        module = ctypes.c_void_p()
        hip.hipModuleLoad(ctypes.byref(module), co.encode())
        func = ctypes.c_void_p()
        hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode())
        funcs[name] = func
        modules[name] = module
    
    torch.manual_seed(42)
    K = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    # Verify correctness
    print("\n1. Correctness check (K=Q=1)")
    print("-" * 60)
    K_uniform = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q_uniform = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    for name, func in funcs.items():
        args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(K_uniform.data_ptr()), ctypes.c_void_p(Q_uniform.data_ptr())]
        args_ptrs = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
        hip.hipDeviceSynchronize()
        result = out.mean().item()
        status = "✓" if abs(result - 128.0) < 0.1 else "✗"
        print(f"   {name:20s}: {result:.2f} {status}")
    
    # Benchmark
    print("\n2. Performance comparison")
    print("-" * 60)
    print(f"   {'Blocks':<8} {'stride128':>12} {'stride132_mul':>14} {'stride132_fast':>15}")
    print("-" * 60)
    
    ops_per_block = 8 * 32 * 32 * 16 * 2
    
    for num_blocks in [64, 256, 512, 1024, 2048, 4096]:
        times = {}
        for name, func in funcs.items():
            times[name] = benchmark(hip, func, K, Q, out, num_blocks, warmup=100, iterations=500)
        
        print(f"   {num_blocks:<8} {times['stride128']:>10.2f}µs {times['stride132_mul']:>12.2f}µs {times['stride132_fast']:>13.2f}µs")
    
    # TF/s comparison at scale
    print("\n3. Throughput at 4096 blocks")
    print("-" * 60)
    
    num_blocks = 4096
    total_ops = ops_per_block * num_blocks
    
    for name, func in funcs.items():
        t = benchmark(hip, func, K, Q, out, num_blocks, warmup=200, iterations=1000)
        tflops = total_ops / (t * 1e-6) / 1e12
        print(f"   {name:20s}: {t:8.2f} µs, {tflops:6.2f} TF/s")
    
    # Speedup analysis
    print("\n4. Speedup analysis")
    print("-" * 60)
    
    t_128 = benchmark(hip, funcs['stride128'], K, Q, out, 4096, warmup=200, iterations=1000)
    t_132_mul = benchmark(hip, funcs['stride132_mul'], K, Q, out, 4096, warmup=200, iterations=1000)
    t_132_fast = benchmark(hip, funcs['stride132_fast'], K, Q, out, 4096, warmup=200, iterations=1000)
    
    print(f"   stride132_mul  vs stride128: {t_128/t_132_mul:.3f}x")
    print(f"   stride132_fast vs stride128: {t_128/t_132_fast:.3f}x")
    print(f"   stride132_fast vs stride132_mul: {t_132_mul/t_132_fast:.3f}x")
    
    for module in modules.values():
        hip.hipModuleUnload(module)
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("  - stride128: 15 bank conflicts per phase, but fast (shift) addressing")
    print("  - stride132_mul: 0 bank conflicts, but slow (multiply) addressing")  
    print("  - stride132_fast: 0 bank conflicts, fast (shift+add) addressing")
    print("=" * 80)

if __name__ == "__main__":
    main()

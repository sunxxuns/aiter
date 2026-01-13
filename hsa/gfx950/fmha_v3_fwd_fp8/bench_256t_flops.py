#!/usr/bin/env python3
"""Benchmark 256T with 4 tiles (proper 4x work)"""

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

def benchmark(hip, func, args_ptrs, num_blocks, threads, warmup=100, iterations=500):
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
    print("BENCHMARK: 64T vs 256T (4 tiles) FLOPS")
    print("=" * 70)
    
    co_64t = build("qk_fp8_stride128")
    co_256t = build("qk_fp8_256t_4tile")
    
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
    hip.hipModuleGetFunction(ctypes.byref(func_256t), module_256t, b"_ZN5aiter17qk_fp8_256t_4tileE")
    
    # Data for 64T: K[32×128], Q[32×128], out[64×16]
    torch.manual_seed(42)
    K_64 = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q_64 = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out_64 = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    # Data for 256T: K[128×128], Q[32×128], out[4×64×16]
    K_256 = torch.randn(128, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q_256 = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out_256 = torch.zeros(4 * 64 * 16, dtype=torch.float32, device='cuda')
    
    args_64 = [ctypes.c_void_p(out_64.data_ptr()), ctypes.c_void_p(K_64.data_ptr()), ctypes.c_void_p(Q_64.data_ptr())]
    args_ptrs_64 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_64])
    
    args_256 = [ctypes.c_void_p(out_256.data_ptr()), ctypes.c_void_p(K_256.data_ptr()), ctypes.c_void_p(Q_256.data_ptr())]
    args_ptrs_256 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_256])
    
    # Correctness check
    print("\n1. Correctness (uniform input)")
    print("-" * 50)
    
    K_u64 = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q_u64 = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K_u256 = torch.ones(128, 128, dtype=torch.float8_e4m3fn, device='cuda')
    Q_u256 = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    args_u64 = [ctypes.c_void_p(out_64.data_ptr()), ctypes.c_void_p(K_u64.data_ptr()), ctypes.c_void_p(Q_u64.data_ptr())]
    args_ptrs_u64 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_u64])
    
    args_u256 = [ctypes.c_void_p(out_256.data_ptr()), ctypes.c_void_p(K_u256.data_ptr()), ctypes.c_void_p(Q_u256.data_ptr())]
    args_ptrs_u256 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_u256])
    
    hip.hipModuleLaunchKernel(func_64t, 1, 1, 1, 64, 1, 1, 32768, None, args_ptrs_u64, None)
    hip.hipDeviceSynchronize()
    print(f"   64T:  mean = {out_64.mean().item():.2f} (expected 128.0)")
    
    hip.hipModuleLaunchKernel(func_256t, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs_u256, None)
    hip.hipDeviceSynchronize()
    print(f"   256T: mean = {out_256.mean().item():.2f} (expected 128.0)")
    
    # FLOPS calculation
    # 64T: 1 wave, 8 MFMAs, each 32×32×16 = 32768 FP8 FMAs = 65536 ops
    ops_64t = 8 * 32 * 32 * 16 * 2  # 262144 ops per block
    
    # 256T: 4 waves, each does 8 MFMAs = 32 MFMAs total
    ops_256t = 4 * 8 * 32 * 32 * 16 * 2  # 1048576 ops per block (4x more)
    
    print(f"\n   Ops per block: 64T={ops_64t:,}, 256T={ops_256t:,} (4x)")
    
    # Benchmark
    print("\n2. Performance")
    print("-" * 50)
    print(f"   {'Blocks':<8} {'64T Time':<12} {'256T Time':<12} {'64T TF/s':<12} {'256T TF/s':<12}")
    print("-" * 70)
    
    for num_blocks in [256, 512, 1024, 2048, 4096]:
        t_64 = benchmark(hip, func_64t, args_ptrs_64, num_blocks, 64)
        t_256 = benchmark(hip, func_256t, args_ptrs_256, num_blocks, 256)
        
        tflops_64 = (ops_64t * num_blocks) / (t_64 * 1e-6) / 1e12
        tflops_256 = (ops_256t * num_blocks) / (t_256 * 1e-6) / 1e12
        
        print(f"   {num_blocks:<8} {t_64:>8.2f} µs   {t_256:>8.2f} µs   {tflops_64:>8.1f}       {tflops_256:>8.1f}")
    
    # Extended benchmark at scale
    print("\n3. Extended benchmark (1000 iterations)")
    print("-" * 50)
    
    for num_blocks in [1024, 2048, 4096]:
        t_64 = benchmark(hip, func_64t, args_ptrs_64, num_blocks, 64, warmup=200, iterations=1000)
        t_256 = benchmark(hip, func_256t, args_ptrs_256, num_blocks, 256, warmup=200, iterations=1000)
        
        tflops_64 = (ops_64t * num_blocks) / (t_64 * 1e-6) / 1e12
        tflops_256 = (ops_256t * num_blocks) / (t_256 * 1e-6) / 1e12
        
        print(f"   {num_blocks} blocks: 64T={tflops_64:.1f} TF/s, 256T={tflops_256:.1f} TF/s")
    
    hip.hipModuleUnload(module_64t)
    hip.hipModuleUnload(module_256t)
    
    print("\n" + "=" * 70)
    print("Note: 256T does 4x the work (4 output tiles vs 1)")
    print("=" * 70)

if __name__ == "__main__":
    main()

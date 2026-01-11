#!/usr/bin/env python3
"""
Performance Benchmark for FP8 Flash Attention HD=128

Measures:
- Kernel latency (microseconds)
- Throughput (GFLOP/s and TF/s)
- Comparison with theoretical peak

Note: Current kernel only supports seq_len=32 (single tile)
Full benchmark at production shapes requires Phase 3 (K-tile loop)
"""

import torch
import subprocess
import ctypes
import os
import time
import math

def build_kernel():
    """Build the kernel"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "test_full_hd128.s", "-o", "test_full_hd128.o"],
        check=True, capture_output=True, cwd=cwd
    )
    subprocess.run(
        ["ld.lld", "-shared", "-o", "test_full_hd128.co", "test_full_hd128.o"],
        check=True, capture_output=True, cwd=cwd
    )
    return os.path.join(cwd, "test_full_hd128.co")


def get_kernel_func(co_path):
    """Load kernel"""
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    func = ctypes.c_void_p()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter14test_full_hd128E")
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return hip, module, func


def benchmark_kernel(hip, func, warmup=10, iterations=100):
    """Benchmark kernel latency"""
    SEQ, HD = 32, 128
    
    # Create test data
    torch.manual_seed(42)
    Q = torch.randn(SEQ, HD, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(SEQ, HD, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    V = torch.randn(SEQ, HD, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(SEQ, HD, dtype=torch.float32, device='cuda')
    
    args_list = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
    ]
    args_array = (ctypes.c_void_p * 4)(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_list]
    )
    
    # Warmup
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_array, None)
    hip.hipDeviceSynchronize()
    
    # Benchmark with HIP events
    hipEventCreate = hip.hipEventCreate
    hipEventRecord = hip.hipEventRecord
    hipEventSynchronize = hip.hipEventSynchronize
    hipEventElapsedTime = hip.hipEventElapsedTime
    hipEventDestroy = hip.hipEventDestroy
    
    start_event = ctypes.c_void_p()
    end_event = ctypes.c_void_p()
    hipEventCreate(ctypes.byref(start_event))
    hipEventCreate(ctypes.byref(end_event))
    
    hipEventRecord(start_event, None)
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_array, None)
    hipEventRecord(end_event, None)
    hipEventSynchronize(end_event)
    
    elapsed_ms = ctypes.c_float()
    hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, end_event)
    
    hipEventDestroy(start_event)
    hipEventDestroy(end_event)
    
    avg_us = (elapsed_ms.value * 1000) / iterations
    
    return avg_us


def compute_flops(seq_len, head_dim):
    """Compute FLOPs for flash attention"""
    # QK: S = Q @ K^T → 2 * seq * seq * head_dim
    qk_flops = 2 * seq_len * seq_len * head_dim
    # Softmax: ~5 ops per element (sub, exp, sum, div, etc.)
    softmax_flops = 5 * seq_len * seq_len
    # PV: O = P @ V → 2 * seq * head_dim * seq
    pv_flops = 2 * seq_len * head_dim * seq_len
    
    total_flops = qk_flops + softmax_flops + pv_flops
    return total_flops


def main():
    print("="*70)
    print("FP8 FLASH ATTENTION BENCHMARK")
    print("HD=128, SEQ=32 (single tile)")
    print("="*70)
    
    # Build and load kernel
    print("\nBuilding kernel...")
    co_path = build_kernel()
    hip, module, func = get_kernel_func(co_path)
    print("Kernel loaded")
    
    # Run benchmark
    SEQ, HD = 32, 128
    print(f"\nBenchmarking (warmup=10, iterations=100)...")
    
    latency_us = benchmark_kernel(hip, func)
    flops = compute_flops(SEQ, HD)
    
    # Calculate throughput
    gflops = (flops / latency_us) / 1000  # GFLOP/s
    tflops = gflops / 1000  # TF/s
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"  Shape: seq={SEQ}, head_dim={HD}")
    print(f"  Latency: {latency_us:.2f} μs")
    print(f"  FLOPs per call: {flops:,}")
    print(f"  Throughput: {gflops:.1f} GFLOP/s ({tflops:.4f} TF/s)")
    
    # MI300X theoretical peak
    # FP8 MFMA: ~2600 TF/s theoretical peak per chip
    # But our kernel is tiny (32x32) so launch overhead dominates
    print(f"\n  Note: Small shape (32×32) - launch overhead dominates")
    print(f"  Production shapes (seq=4096+) needed for meaningful TF/s")
    
    # Estimate at larger shapes (theoretical)
    print(f"\n{'='*70}")
    print("PROJECTED PERFORMANCE (theoretical, requires K-tile loop)")
    print('='*70)
    
    for seq in [1024, 4096, 8192, 16384, 32768]:
        # Assume kernel time scales linearly with tiles (optimistic)
        num_tiles = (seq // 32) ** 2  # K-tiles × Q-tiles
        est_time_us = latency_us * num_tiles
        est_flops = compute_flops(seq, HD)
        est_tflops = (est_flops / est_time_us) / 1e9
        
        # More realistic: add some overhead per tile
        overhead_per_tile_us = 0.5
        realistic_time_us = latency_us * num_tiles + overhead_per_tile_us * num_tiles
        realistic_tflops = (est_flops / realistic_time_us) / 1e9
        
        print(f"  seq={seq:5d}: {num_tiles:6d} tiles, est {est_tflops:.1f}-{realistic_tflops:.1f} TF/s")
    
    # Compare to BF16 baseline
    print(f"\n{'='*70}")
    print("BF16 BASELINE COMPARISON (from bench_aiter_fmha_v3.py)")
    print('='*70)
    print("  BF16 ASM kernel at similar shapes:")
    print("    seq=1024:  ~400 TF/s")
    print("    seq=4096:  ~780 TF/s")
    print("    seq=8192:  ~980 TF/s")
    print("    seq=32130: ~1000 TF/s")
    print(f"\n  FP8 Target: >1300 TF/s (30%+ improvement)")
    
    hip.hipModuleUnload(module)
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print('='*70)
    print("  Current kernel: Single tile (32×32) - for numerical validation only")
    print("  Next step: Implement K-tile loop (Phase 3) for production benchmark")
    print('='*70)


if __name__ == "__main__":
    main()

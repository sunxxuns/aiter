#!/usr/bin/env python3
"""
FP8 Flash Attention Benchmark

Benchmarks the FP8 ASM kernel (fwd_fp8_kloop.s) at various sequence lengths.
Includes BF16 single-head comparison for fair apples-to-apples comparison.

Usage:
    python bench_fp8_attention.py                    # Default seq_len=1024
    python bench_fp8_attention.py --seq-len 4096    # Specific seq_len
    python bench_fp8_attention.py --sweep           # Sweep all seq_lens
    python bench_fp8_attention.py --compare         # Compare FP8 vs BF16
"""

import torch
import subprocess
import ctypes
import os
import argparse
import math
import time


def build_kernel(kernel_name="fwd_fp8_kloop"):
    """Build the FP8 kernel"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    src = f"{kernel_name}.s"
    obj = f"{kernel_name}.o"
    co = f"{kernel_name}.co"
    
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", src, "-o", obj],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compile failed: {result.stderr.decode()}")
    
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", co, obj],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"Link failed: {result.stderr.decode()}")
    
    return os.path.join(cwd, co)


def load_kernel(co_path, func_name="_ZN5aiter13fwd_fp8_kloopE"):
    """Load kernel module"""
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    
    func = ctypes.c_void_p()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    
    return hip, module, func


def benchmark_fp8_fmha(
    seq_len: int = 1024,
    head_dim: int = 128,
    warmup: int = 10,
    iters: int = 100,
):
    """Benchmark FP8 Flash Attention kernel."""
    
    # Build and load
    co_path = build_kernel()
    hip, module, func = load_kernel(co_path)
    
    # Create test data (single head, single batch for now)
    Q_ROWS = 32  # Output rows per workgroup
    
    torch.manual_seed(42)
    Q = torch.randn(Q_ROWS, head_dim, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    V = torch.randn(seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(Q_ROWS, head_dim, dtype=torch.float32, device='cuda')
    
    # Kernel args
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_uint32(seq_len),
    ]
    args_arr = (ctypes.c_void_p * 5)(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    
    shared_mem = 12288  # 12KB LDS
    
    # Warmup
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, shared_mem, None, args_arr, None)
    hip.hipDeviceSynchronize()
    
    # Benchmark with HIP events
    start_event = ctypes.c_void_p()
    end_event = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start_event))
    hip.hipEventCreate(ctypes.byref(end_event))
    
    hip.hipEventRecord(start_event, None)
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, shared_mem, None, args_arr, None)
    hip.hipEventRecord(end_event, None)
    hip.hipEventSynchronize(end_event)
    
    elapsed_ms = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, end_event)
    
    hip.hipEventDestroy(start_event)
    hip.hipEventDestroy(end_event)
    
    avg_time_ms = elapsed_ms.value / iters
    
    # Calculate metrics
    # FLOPs: QK (2*Q*K*D) + softmax (~5*Q*K) + PV (2*Q*K*D)
    flops = Q_ROWS * seq_len * head_dim * 4 + Q_ROWS * seq_len * 5
    tflops = (flops / avg_time_ms) / 1e9  # TFLOPs
    
    # Memory: Q + K + V + O
    mem_bytes = Q_ROWS * head_dim + seq_len * head_dim * 2 + Q_ROWS * head_dim * 4  # FP8 + F32
    bandwidth = (mem_bytes / avg_time_ms) / 1e6  # GB/s
    
    hip.hipModuleUnload(module)
    
    return {
        'seq_len': seq_len,
        'time_ms': avg_time_ms,
        'time_us': avg_time_ms * 1000,
        'tflops': tflops,
        'bandwidth_gbs': bandwidth,
        'flops': flops,
    }


def benchmark_bf16_fmha(
    seq_len: int = 1024,
    head_dim: int = 128,
    num_heads: int = 1,
    warmup: int = 10,
    iters: int = 100,
):
    """Benchmark BF16 Flash Attention kernel (single head for fair comparison)."""
    from aiter.ops.mha import fmha_v3_fwd
    
    # Match FP8 kernel shape: single head, full seq_len
    q = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    def run():
        return fmha_v3_fwd(q, k, v, 0.0, softmax_scale, False, -1, -1, True, False, 1)
    
    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    avg_time_ms = sum(times) / len(times)
    
    # Calculate metrics (same FLOP formula as FP8)
    # For fair comparison, use same formula: 32 rows * seq_len
    # But BF16 kernel processes full seq_len x seq_len
    flops = 4 * 1 * num_heads * seq_len * seq_len * head_dim
    tflops = (flops / avg_time_ms) / 1e9
    
    return {
        'seq_len': seq_len,
        'time_ms': avg_time_ms,
        'time_us': avg_time_ms * 1000,
        'tflops': tflops,
        'flops': flops,
    }


def main():
    parser = argparse.ArgumentParser(description="FP8 Flash Attention Benchmark")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--sweep", action="store_true", help="Sweep sequence lengths")
    parser.add_argument("--compare", action="store_true", help="Compare FP8 vs BF16")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()
    
    if args.compare:
        print(f"\n{'='*90}")
        print("FP8 vs BF16 Flash Attention - Single Head Comparison")
        print(f"{'='*90}")
        print(f"FP8:  32 Q-rows × seq_len K-rows (single workgroup, K-loop)")
        print(f"BF16: seq_len Q-rows × seq_len K-rows (many workgroups, fully optimized)")
        print(f"{'='*90}")
        
        # Table 1: Raw kernel times
        print(f"\n1. RAW KERNEL TIMES (total work differs)")
        print(f"{'-'*90}")
        print(f"{'SeqLen':>8} {'FP8 32×N':>12} {'BF16 N×N':>12} {'FP8 FLOPs':>14} {'BF16 FLOPs':>14}")
        print(f"{'':>8} {'(us)':>12} {'(us)':>12} {'':>14} {'':>14}")
        print(f"{'-'*90}")
        
        results = []
        for seq_len in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
            try:
                fp8_result = benchmark_fp8_fmha(seq_len=seq_len, iters=args.iters)
                bf16_result = benchmark_bf16_fmha(seq_len=seq_len, iters=args.iters)
                results.append((seq_len, fp8_result, bf16_result))
                
                print(f"{seq_len:>8} {fp8_result['time_us']:>12.1f} {bf16_result['time_us']:>12.1f} "
                      f"{fp8_result['flops']:>14,} {bf16_result['flops']:>14,}")
            except Exception as e:
                print(f"{seq_len:>8} ERROR: {e}")
        
        # Table 2: Normalized comparison (time per GFLOP)
        print(f"\n2. EFFICIENCY (time per GFLOP - lower is better)")
        print(f"{'-'*90}")
        print(f"{'SeqLen':>8} {'FP8 us/GF':>12} {'BF16 us/GF':>12} {'FP8 Eff':>10} {'Speedup':>10}")
        print(f"{'-'*90}")
        
        for seq_len, fp8_result, bf16_result in results:
            fp8_us_per_gflop = fp8_result['time_us'] / (fp8_result['flops'] / 1e9)
            bf16_us_per_gflop = bf16_result['time_us'] / (bf16_result['flops'] / 1e9)
            speedup = bf16_us_per_gflop / fp8_us_per_gflop
            
            # Efficiency relative to BF16 at same seq_len
            eff_pct = (bf16_us_per_gflop / fp8_us_per_gflop) * 100
            
            print(f"{seq_len:>8} {fp8_us_per_gflop:>12.3f} {bf16_us_per_gflop:>12.3f} "
                  f"{eff_pct:>9.1f}% {speedup:>9.2f}x")
        
        print(f"{'='*90}")
        print(f"\nInterpretation:")
        print(f"  - 'us/GF': microseconds per GigaFLOP (lower = better)")
        print(f"  - 'FP8 Eff': FP8 efficiency relative to BF16 (>100% = FP8 faster)")
        print(f"  - 'Speedup': FP8 speedup over BF16 (>1.0 = FP8 faster)")
        print(f"\nNote: BF16 benefits from multi-workgroup parallelism at large seq_len.")
        print(f"      FP8 needs Phase 5+ (multi-head/multi-tile) to match.")
        
    elif args.sweep:
        print(f"\n{'='*70}")
        print("FP8 Flash Attention - Sequence Length Sweep")
        print(f"{'='*70}")
        print(f"{'SeqLen':>8} {'Tiles':>6} {'Time(us)':>10} {'TFLOPS':>10}")
        print(f"{'-'*70}")
        
        for seq_len in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
            try:
                result = benchmark_fp8_fmha(seq_len=seq_len, iters=args.iters)
                tiles = (seq_len + 31) // 32
                print(f"{seq_len:>8} {tiles:>6} {result['time_us']:>10.1f} {result['tflops']:>10.4f}")
            except Exception as e:
                print(f"{seq_len:>8} ERROR: {e}")
        
        print(f"{'='*70}")
        
    else:
        print(f"\n{'='*70}")
        print(f"FP8 Flash Attention Benchmark")
        print(f"{'='*70}")
        
        result = benchmark_fp8_fmha(seq_len=args.seq_len, iters=args.iters)
        
        print(f"Seq Length:  {result['seq_len']}")
        print(f"K-tiles:     {(result['seq_len'] + 31) // 32}")
        print(f"Time:        {result['time_us']:.1f} μs ({result['time_ms']:.3f} ms)")
        print(f"TFLOPS:      {result['tflops']:.4f}")
        print(f"Bandwidth:   {result['bandwidth_gbs']:.1f} GB/s")
        print(f"{'='*70}")
        
        # Comparison note
        print(f"\nNote: This is single-head, 32-row output.")
        print(f"Use --compare to see FP8 vs BF16 comparison.")


if __name__ == "__main__":
    main()

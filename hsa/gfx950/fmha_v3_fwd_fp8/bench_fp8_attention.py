#!/usr/bin/env python3
"""
FP8 Flash Attention Benchmark

Benchmarks the FP8 ASM kernel (fwd_fp8_kloop.s) at various sequence lengths.
Comparable to BF16 benchmark: bench_mi350_fmha_asm.py

Usage:
    python bench_fp8_attention.py                    # Default seq_len=1024
    python bench_fp8_attention.py --seq-len 4096    # Specific seq_len
    python bench_fp8_attention.py --sweep           # Sweep all seq_lens
"""

import torch
import subprocess
import ctypes
import os
import argparse
import math


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


def main():
    parser = argparse.ArgumentParser(description="FP8 Flash Attention Benchmark")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--sweep", action="store_true", help="Sweep sequence lengths")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()
    
    if args.sweep:
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
        print(f"Full benchmark requires multi-head kernel (Phase 5+)")


if __name__ == "__main__":
    main()

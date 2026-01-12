#!/usr/bin/env python3
"""
Benchmark FP8 Full Attention vs BF16

Now comparing apples-to-apples: both do full seq×seq attention.
"""

import torch
import subprocess
import ctypes
import time
import argparse

def build_fp8():
    """Build FP8 full attention kernel"""
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "fwd_fp8_full.s", "-o", "fwd_fp8_full.o"],
        capture_output=True, cwd=cwd, check=True
    )
    subprocess.run(
        ["ld.lld", "-shared", "-o", "fwd_fp8_full.co", "fwd_fp8_full.o"],
        capture_output=True, cwd=cwd, check=True
    )
    return cwd + "/fwd_fp8_full.co"


def benchmark_fp8(seq_len, warmup=10, iters=100):
    """Benchmark FP8 full attention"""
    co_path = build_fp8()
    
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12fwd_fp8_fullE")
    
    HD = 128
    num_q_tiles = seq_len // 32
    
    Q = torch.randn(seq_len, HD, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(seq_len, HD, device='cuda').to(torch.float8_e4m3fn)
    V = torch.randn(seq_len, HD, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(seq_len, HD, dtype=torch.float32, device='cuda')
    
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
    
    # Warmup
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, num_q_tiles, 1, 1, 64, 1, 1, 12288, None, args_arr, None)
    hip.hipDeviceSynchronize()
    
    # Benchmark
    start_event = ctypes.c_void_p()
    end_event = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start_event))
    hip.hipEventCreate(ctypes.byref(end_event))
    
    hip.hipEventRecord(start_event, None)
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func, num_q_tiles, 1, 1, 64, 1, 1, 12288, None, args_arr, None)
    hip.hipEventRecord(end_event, None)
    hip.hipEventSynchronize(end_event)
    
    elapsed_ms = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, end_event)
    
    hip.hipEventDestroy(start_event)
    hip.hipEventDestroy(end_event)
    hip.hipModuleUnload(module)
    
    avg_ms = elapsed_ms.value / iters
    
    # FLOPs: QK (2*seq*seq*HD) + softmax (~5*seq*seq) + PV (2*seq*seq*HD)
    flops = seq_len * seq_len * HD * 4 + seq_len * seq_len * 5
    tflops = (flops / avg_ms) / 1e9
    
    return {'time_ms': avg_ms, 'time_us': avg_ms * 1000, 'tflops': tflops}


def benchmark_bf16(seq_len, warmup=10, iters=100):
    """Benchmark BF16 attention using aiter API"""
    from aiter.ops.mha import fmha_v3_fwd
    
    HD = 128
    q = torch.randn(1, seq_len, 1, HD, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(1, seq_len, 1, HD, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(1, seq_len, 1, HD, dtype=torch.bfloat16, device='cuda')
    
    softmax_scale = 1.0 / (HD ** 0.5)
    
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
        times.append((time.perf_counter() - start) * 1000)
    
    avg_ms = sum(times) / len(times)
    
    flops = seq_len * seq_len * HD * 4 + seq_len * seq_len * 5
    tflops = (flops / avg_ms) / 1e9
    
    return {'time_ms': avg_ms, 'time_us': avg_ms * 1000, 'tflops': tflops}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 80)
    print("FP8 vs BF16 FULL ATTENTION BENCHMARK")
    print("=" * 80)
    print("Both kernels now do FULL seq×seq attention (apples-to-apples)")
    print("=" * 80)
    
    if args.sweep:
        print(f"\n{'SeqLen':>8} {'FP8(us)':>12} {'BF16(us)':>12} {'FP8 TF/s':>10} {'BF16 TF/s':>10} {'Speedup':>10}")
        print("-" * 80)
        
        for seq_len in [64, 128, 256, 512, 1024, 2048]:
            fp8 = benchmark_fp8(seq_len, iters=args.iters)
            bf16 = benchmark_bf16(seq_len, iters=args.iters)
            
            speedup = bf16['time_us'] / fp8['time_us']
            
            print(f"{seq_len:>8} {fp8['time_us']:>12.1f} {bf16['time_us']:>12.1f} "
                  f"{fp8['tflops']:>10.4f} {bf16['tflops']:>10.4f} {speedup:>9.2f}x")
        
        print("=" * 80)
        print("\nSpeedup > 1.0 means FP8 is faster")
        print("Target: FP8 should be 1.3x+ faster (30% improvement)")
        
    else:
        seq_len = args.seq_len
        print(f"\nSeq length: {seq_len}")
        print(f"Q-tiles: {seq_len // 32}")
        
        print("\nBenchmarking FP8...")
        fp8 = benchmark_fp8(seq_len, iters=args.iters)
        
        print("Benchmarking BF16...")
        bf16 = benchmark_bf16(seq_len, iters=args.iters)
        
        print(f"\n{'Metric':<20} {'FP8':>15} {'BF16':>15} {'Ratio':>10}")
        print("-" * 60)
        print(f"{'Time (us)':<20} {fp8['time_us']:>15.1f} {bf16['time_us']:>15.1f} {bf16['time_us']/fp8['time_us']:>9.2f}x")
        print(f"{'TF/s':<20} {fp8['tflops']:>15.4f} {bf16['tflops']:>15.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()

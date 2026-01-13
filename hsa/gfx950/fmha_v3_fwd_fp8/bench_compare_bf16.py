#!/usr/bin/env python3
"""
Compare FP8 QK microbenchmark vs BF16 full flash attention
to understand the performance gap
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
    print("FP8 QK Microbenchmark vs BF16 Full Flash Attention")
    print("=" * 70)
    
    # Build FP8 kernels
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
    
    # FP8 data
    K_64 = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q_64 = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out_64 = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
    
    K_256 = torch.randn(128, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    Q_256 = torch.randn(32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    out_256 = torch.zeros(4 * 64 * 16, dtype=torch.float32, device='cuda')
    
    args_64 = [ctypes.c_void_p(out_64.data_ptr()), ctypes.c_void_p(K_64.data_ptr()), ctypes.c_void_p(Q_64.data_ptr())]
    args_ptrs_64 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_64])
    
    args_256 = [ctypes.c_void_p(out_256.data_ptr()), ctypes.c_void_p(K_256.data_ptr()), ctypes.c_void_p(Q_256.data_ptr())]
    args_ptrs_256 = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_256])
    
    # FLOPS definitions
    ops_64t = 8 * 32 * 32 * 16 * 2   # 262,144
    ops_256t = 4 * ops_64t            # 1,048,576
    
    print("\n1. BF16 Full Flash Attention (from benchmark)")
    print("-" * 50)
    print("   seq_len=1024:  409 TF/s")
    print("   seq_len=4096:  869 TF/s")
    print("   seq_len=32130: 1018 TF/s")
    
    print("\n2. FP8 QK Microbenchmark")
    print("-" * 50)
    
    # Saturate the GPU with many blocks
    for num_blocks in [4096, 8192, 16384]:
        t_64 = benchmark(hip, func_64t, args_ptrs_64, num_blocks, 64, warmup=100, iterations=500)
        t_256 = benchmark(hip, func_256t, args_ptrs_256, num_blocks, 256, warmup=100, iterations=500)
        
        tflops_64 = (ops_64t * num_blocks) / (t_64 * 1e-6) / 1e12
        tflops_256 = (ops_256t * num_blocks) / (t_256 * 1e-6) / 1e12
        
        print(f"   {num_blocks} blocks: 64T={tflops_64:.1f} TF/s, 256T={tflops_256:.1f} TF/s")
    
    print("\n3. Analysis")
    print("-" * 50)
    
    # Best FP8 result
    t_best = benchmark(hip, func_256t, args_ptrs_256, 16384, 256, warmup=200, iterations=1000)
    tflops_best = (ops_256t * 16384) / (t_best * 1e-6) / 1e12
    
    print(f"   FP8 QK best:     {tflops_best:.1f} TF/s")
    print(f"   BF16 FA best:    1018 TF/s")
    print(f"   FP8/BF16 ratio:  {tflops_best/1018*100:.1f}%")
    print()
    print("   Note: FP8 QK is microbenchmark (QK only)")
    print("         BF16 FA is full flash attention (QK + softmax + PV)")
    print()
    print("   For fair comparison, FP8 should be ~2x faster than BF16")
    print("   because FP8 MFMA has 2x throughput of BF16 MFMA")
    
    hip.hipModuleUnload(module_64t)
    hip.hipModuleUnload(module_256t)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark 256T FP8 flash attention kernel"""

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
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', f'{name}.co', f'{name}.o'], cwd=cwd)
    return str(cwd / f'{name}.co')

def main():
    print("=" * 70)
    print("Benchmark: 64T vs 256T FP8 Flash Attention")
    print("=" * 70)
    
    co_64t = build("fwd_fp8_kloop")
    co_256t = build("fwd_fp8_256t")
    
    if not co_64t or not co_256t:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module_64t = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_64t), co_64t.encode())
    func_64t = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_64t), module_64t, b"_ZN5aiter13fwd_fp8_kloopE")
    
    module_256t = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module_256t), co_256t.encode())
    func_256t = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func_256t), module_256t, b"_ZN5aiter13fwd_fp8_256tE")
    
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start))
    hip.hipEventCreate(ctypes.byref(end))
    
    seq_len = 1024
    
    # Data
    O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    Q = torch.ones(32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda')
    V = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_uint32(seq_len)
    ]
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    print(f"\nseq_len = {seq_len}, K-tiles = {(seq_len+31)//32}")
    print("-" * 70)
    
    # Test 64T
    print("\n64T (1 wave):")
    for num_blocks in [1, 40, 160]:
        # Warmup
        for _ in range(10):
            hip.hipModuleLaunchKernel(func_64t, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
        hip.hipDeviceSynchronize()
        
        # Benchmark
        iterations = 50
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            hip.hipModuleLaunchKernel(func_64t, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
        hip.hipEventRecord(end, None)
        hip.hipEventSynchronize(end)
        
        elapsed = ctypes.c_float()
        hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
        time_us = elapsed.value * 1000 / iterations
        
        flops = 4 * 32 * seq_len * 128 * num_blocks
        tflops = flops / (time_us * 1e-6) / 1e12
        
        print(f"  {num_blocks} blocks: {time_us:.2f} µs, {tflops:.2f} TF/s")
    
    # Test 256T
    print("\n256T (4 waves):")
    for num_blocks in [1, 40, 160]:
        # Warmup
        for _ in range(10):
            hip.hipModuleLaunchKernel(func_256t, num_blocks, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
        hip.hipDeviceSynchronize()
        
        # Benchmark
        iterations = 50
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            hip.hipModuleLaunchKernel(func_256t, num_blocks, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
        hip.hipEventRecord(end, None)
        hip.hipEventSynchronize(end)
        
        elapsed = ctypes.c_float()
        hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
        time_us = elapsed.value * 1000 / iterations
        
        flops = 4 * 32 * seq_len * 128 * num_blocks
        tflops = flops / (time_us * 1e-6) / 1e12
        
        print(f"  {num_blocks} blocks: {time_us:.2f} µs, {tflops:.2f} TF/s")
    
    # Correctness check
    print("\nCorrectness check (uniform K=Q=V=1):")
    O.zero_()
    hip.hipModuleLaunchKernel(func_256t, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    print(f"  O mean: {O.mean().item():.4f}")
    print(f"  O[0,0:4]: {O[0,0:4].tolist()}")
    
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(end)
    hip.hipModuleUnload(module_64t)
    hip.hipModuleUnload(module_256t)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

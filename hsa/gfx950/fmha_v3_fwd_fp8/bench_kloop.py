#!/usr/bin/env python3
"""Benchmark fwd_fp8_kloop.s - the full FP8 flash attention kernel"""

import torch
import subprocess
import ctypes
import time
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'fwd_fp8_kloop.s', '-o', 'fwd_fp8_kloop.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'fwd_fp8_kloop.co', 'fwd_fp8_kloop.o'], cwd=cwd)
    return str(cwd / 'fwd_fp8_kloop.co')

def main():
    print("=" * 70)
    print("Benchmark: fwd_fp8_kloop.s (Full FP8 Flash Attention)")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13fwd_fp8_kloopE")
    
    start_event = ctypes.c_void_p()
    end_event = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start_event))
    hip.hipEventCreate(ctypes.byref(end_event))
    
    print("\nFP8 Flash Attention (32×128 Q tile, seq=1024)")
    print("-" * 70)
    print(f"{'Blocks':<10} {'seq_len':<10} {'Time (µs)':<12} {'TF/s':<10}")
    print("-" * 70)
    
    # BF16 reference times from earlier benchmarks
    bf16_tflops = {128: 409, 512: 700, 1024: 869, 4096: 1000, 32130: 1018}
    
    # Test with multiple blocks (like multiple attention heads)
    for num_blocks in [1, 40, 160, 640]:
        seq_len = 1024
        
        # Data for all blocks
        O = torch.zeros(num_blocks, 32, 128, dtype=torch.float32, device='cuda')
        Q = torch.randn(num_blocks, 32, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        K = torch.randn(num_blocks, seq_len, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        V = torch.randn(num_blocks, seq_len, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        
        # Each block processes its own data (simplified - using same pointers)
        args = [
            ctypes.c_void_p(O.data_ptr()),
            ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()),
            ctypes.c_void_p(V.data_ptr()),
            ctypes.c_uint32(seq_len)
        ]
        args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        
        # Warmup
        for _ in range(20):
            hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
        hip.hipDeviceSynchronize()
        
        # Benchmark
        iterations = 100
        hip.hipEventRecord(start_event, None)
        for _ in range(iterations):
            hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 64, 1, 1, 32768, None, args_ptrs, None)
        hip.hipEventRecord(end_event, None)
        hip.hipEventSynchronize(end_event)
        
        elapsed = ctypes.c_float()
        hip.hipEventElapsedTime(ctypes.byref(elapsed), start_event, end_event)
        time_us = elapsed.value * 1000 / iterations
        
        # FLOPS: QK (2*32*seq*128) + softmax (~) + PV (2*32*seq*128) ≈ 4*32*seq*128 per block
        flops_per_block = 4 * 32 * seq_len * 128
        total_flops = flops_per_block * num_blocks
        tflops = total_flops / (time_us * 1e-6) / 1e12
        
        k_tiles = (seq_len + 31) // 32
        
        print(f"{num_blocks:<10} {seq_len:<10} {time_us:<12.2f} {tflops:<10.2f}")
    
    hip.hipEventDestroy(start_event)
    hip.hipEventDestroy(end_event)
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)
    print("Note: This is 64T (1 wave) kernel with single 32×128 Q tile")
    print("BF16 uses 256T (4 waves) with multiple Q tiles")
    print("=" * 70)

if __name__ == "__main__":
    main()

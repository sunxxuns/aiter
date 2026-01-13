#!/usr/bin/env python3
"""Test FP8 128Q kernel - 128 Q rows, 256 threads"""

import torch
import subprocess
import ctypes
import math
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'fwd_fp8_128q.s', '-o', 'fwd_fp8_128q.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'fwd_fp8_128q.co', 'fwd_fp8_128q.o'], cwd=cwd)
    return str(cwd / 'fwd_fp8_128q.co')

def reference(Q_fp8, K_fp8, V_fp8):
    """Compute reference output"""
    scale = 1.0 / math.sqrt(128)
    S = Q_fp8.float() @ K_fp8.float().T * scale
    P = torch.softmax(S, dim=1)
    P_fp8 = P.to(torch.float8_e4m3fn).float()
    return P_fp8 @ V_fp8.float()

def main():
    print("=" * 70)
    print("FP8 128Q Kernel Test - 128 Q rows, 256 threads (4 waves)")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13fwd_fp8_128qE")
    
    # Test 1: V=1 identity test
    print("\nTest 1: V=1 identity (output should be ~1.0)")
    print("-" * 50)
    
    seq_len = 128
    Q = torch.randn(128, 128, device='cuda') * 0.5
    K = torch.randn(seq_len, 128, device='cuda') * 0.5
    V = torch.ones(seq_len, 128, device='cuda')
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    O = torch.zeros(128, 128, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(x.data_ptr()) for x in [O, Q_fp8, K_fp8, V_fp8]]
    args.append(ctypes.c_uint32(seq_len))
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    print(f"  O mean: {O.mean().item():.4f} (expected: 1.0)")
    print(f"  O[0,0:4]: {O[0,0:4].tolist()}")
    print(f"  O[64,0:4]: {O[64,0:4].tolist()}")  # Wave 2
    print(f"  Has NaN: {torch.isnan(O).any().item()}")
    
    # Check per-wave results
    for wave in range(4):
        wave_mean = O[wave*32:(wave+1)*32, :].mean().item()
        print(f"  Wave {wave} rows {wave*32}-{(wave+1)*32-1}: mean={wave_mean:.4f}")
    
    # Test 2: Reference comparison
    print("\nTest 2: Reference comparison")
    print("-" * 50)
    
    torch.manual_seed(42)
    Q = torch.randn(128, 128, device='cuda') * 0.5
    K = torch.randn(seq_len, 128, device='cuda') * 0.5
    V = torch.randn(seq_len, 128, device='cuda') * 0.5
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    O = torch.zeros(128, 128, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(x.data_ptr()) for x in [O, Q_fp8, K_fp8, V_fp8]]
    args.append(ctypes.c_uint32(seq_len))
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    O_ref = reference(Q_fp8, K_fp8, V_fp8)
    
    max_err = (O - O_ref).abs().max().item()
    mean_err = (O - O_ref).abs().mean().item()
    
    print(f"  Max error: {max_err:.4f}")
    print(f"  Mean error: {mean_err:.4f}")
    print(f"  O mean: {O.mean().item():.4f}")
    print(f"  O_ref mean: {O_ref.mean().item():.4f}")
    
    # Test 3: Benchmark
    print("\nTest 3: Benchmark")
    print("-" * 50)
    
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start))
    hip.hipEventCreate(ctypes.byref(end))
    
    seq_len = 1024
    K_fp8 = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    V_fp8 = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    
    args = [ctypes.c_void_p(x.data_ptr()) for x in [O, Q_fp8, K_fp8, V_fp8]]
    args.append(ctypes.c_uint32(seq_len))
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Warmup
    for _ in range(10):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Benchmark
    iterations = 100
    hip.hipEventRecord(start, None)
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipEventRecord(end, None)
    hip.hipEventSynchronize(end)
    
    elapsed = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
    time_us = elapsed.value * 1000 / iterations
    
    # FLOPs: QK (128*seq*128*2) + PV (128*seq*128*2) + softmax (~)
    flops = 4 * 128 * seq_len * 128
    tflops = flops / (time_us * 1e-6) / 1e12
    
    print(f"  seq_len={seq_len}, time={time_us:.2f} µs, {tflops:.2f} TF/s")
    
    # Multi-block test
    print("\n  Multi-block scaling (simulating multiple heads):")
    for num_blocks in [1, 10, 40, 100]:
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
        hip.hipEventRecord(end, None)
        hip.hipEventSynchronize(end)
        
        hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end)
        time_us = elapsed.value * 1000 / iterations
        total_flops = flops * num_blocks
        tflops = total_flops / (time_us * 1e-6) / 1e12
        
        print(f"    {num_blocks} blocks: {time_us:.2f} µs, {tflops:.2f} TF/s")
    
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(end)
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

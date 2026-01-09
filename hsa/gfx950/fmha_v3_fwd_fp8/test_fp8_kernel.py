#!/usr/bin/env python3
"""
Test script for FP8 Flash Attention ASM Kernel
"""

import os
import ctypes
import struct
import torch
import argparse

# Load HIP library
libhip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")

hipModule_t = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p

libhip.hipModuleLoad.argtypes = [ctypes.POINTER(hipModule_t), ctypes.c_char_p]
libhip.hipModuleLoad.restype = ctypes.c_int
libhip.hipModuleGetFunction.argtypes = [ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.c_char_p]
libhip.hipModuleGetFunction.restype = ctypes.c_int
libhip.hipModuleLaunchKernel.argtypes = [
    hipFunction_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
libhip.hipModuleLaunchKernel.restype = ctypes.c_int
libhip.hipDeviceSynchronize.argtypes = []
libhip.hipDeviceSynchronize.restype = ctypes.c_int

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP8_CO = os.path.join(SCRIPT_DIR, "fwd_hd128_fp8.co")
KERNEL_NAME = "_ZN5aiter18fmha_fwd_hd128_fp8E"


def pack_args(q, k, v, out, lse, scale, seqlen_q, seqlen_k, q_s, k_s, v_s):
    """Pack kernel arguments."""
    args = bytearray(528)
    struct.pack_into('Q', args, 0, out.data_ptr())
    struct.pack_into('Q', args, 16, q.data_ptr())
    struct.pack_into('Q', args, 32, k.data_ptr())
    struct.pack_into('Q', args, 48, v.data_ptr())
    struct.pack_into('Q', args, 64, lse.data_ptr())
    struct.pack_into('f', args, 80, scale)
    struct.pack_into('I', args, 88, seqlen_q)
    struct.pack_into('I', args, 96, seqlen_k)
    struct.pack_into('f', args, 512, q_s)
    struct.pack_into('f', args, 516, k_s)
    struct.pack_into('f', args, 520, v_s)
    return bytes(args)


def launch_kernel(function, args, grid, block, shared_mem=32768):
    """Launch the kernel."""
    args_ptr = ctypes.cast(args, ctypes.c_void_p)
    args_size = ctypes.c_size_t(len(args))
    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(0x01), args_ptr,
        ctypes.c_void_p(0x02), ctypes.cast(ctypes.pointer(args_size), ctypes.c_void_p),
        ctypes.c_void_p(0x03)
    )
    err = libhip.hipModuleLaunchKernel(
        function, grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        shared_mem, None, None, ctypes.cast(extra, ctypes.c_void_p)
    )
    return err


def test_fp8_kernel(batch=1, seq_len=1024, num_heads=8, head_dim=128, verbose=True):
    """Test the FP8 kernel."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing FP8 Flash Attention Kernel")
        print(f"{'='*60}")
        print(f"Shape: B={batch}, S={seq_len}, H={num_heads}, D={head_dim}")
    
    # Load kernel
    module = hipModule_t()
    err = libhip.hipModuleLoad(ctypes.byref(module), FP8_CO.encode())
    if err != 0:
        print(f"ERROR: hipModuleLoad failed with error {err}")
        return False
    
    function = hipFunction_t()
    err = libhip.hipModuleGetFunction(ctypes.byref(function), module, KERNEL_NAME.encode())
    if err != 0:
        print(f"ERROR: hipModuleGetFunction failed with error {err}")
        return False
    
    if verbose:
        print(f"Kernel loaded successfully: {KERNEL_NAME}")
    
    # Create tensors
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.float32, device='cuda')
    lse = torch.zeros(batch, num_heads, seq_len, dtype=torch.float32, device='cuda')
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    q_scale = k_scale = v_scale = 1.0
    
    args = pack_args(q, k, v, out, lse, softmax_scale, seq_len, seq_len, q_scale, k_scale, v_scale)
    
    # Grid/block dimensions
    BLOCK_M = 64
    grid = ((seq_len + BLOCK_M - 1) // BLOCK_M, num_heads, batch)
    block = (256, 1, 1)
    
    if verbose:
        print(f"Grid: {grid}, Block: {block}")
    
    # Launch kernel
    err = launch_kernel(function, args, grid, block)
    if err != 0:
        print(f"ERROR: Kernel launch failed with error {err}")
        return False
    
    # Synchronize
    err = libhip.hipDeviceSynchronize()
    if err != 0:
        print(f"ERROR: hipDeviceSynchronize failed with error {err}")
        return False
    
    if verbose:
        print(f"Kernel executed successfully!")
        print(f"\nOutput stats:")
        print(f"  Shape: {out.shape}")
        print(f"  Min: {out.min().item():.6f}")
        print(f"  Max: {out.max().item():.6f}")
        print(f"  Mean: {out.mean().item():.6f}")
        print(f"  Std: {out.std().item():.6f}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if not has_nan and not has_inf:
            print(f"\n✓ Test PASSED!")
        else:
            print(f"\n✗ Test FAILED - output contains NaN or Inf")
            return False
    
    return True


def benchmark(batch=1, seq_len=4096, num_heads=40, head_dim=128, warmup=10, iters=50):
    """Benchmark the kernel."""
    import time
    
    print(f"\n{'='*60}")
    print(f"Benchmarking FP8 Flash Attention Kernel")
    print(f"{'='*60}")
    print(f"Shape: B={batch}, S={seq_len}, H={num_heads}, D={head_dim}")
    
    # Load kernel
    module = hipModule_t()
    libhip.hipModuleLoad(ctypes.byref(module), FP8_CO.encode())
    function = hipFunction_t()
    libhip.hipModuleGetFunction(ctypes.byref(function), module, KERNEL_NAME.encode())
    
    # Create tensors
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.float32, device='cuda')
    lse = torch.zeros(batch, num_heads, seq_len, dtype=torch.float32, device='cuda')
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    args = pack_args(q, k, v, out, lse, softmax_scale, seq_len, seq_len, 1.0, 1.0, 1.0)
    
    BLOCK_M = 64
    grid = ((seq_len + BLOCK_M - 1) // BLOCK_M, num_heads, batch)
    block = (256, 1, 1)
    
    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        launch_kernel(function, args, grid, block)
    libhip.hipDeviceSynchronize()
    
    # Benchmark
    print(f"Benchmarking ({iters} iterations)...")
    start = time.perf_counter()
    for _ in range(iters):
        launch_kernel(function, args, grid, block)
    libhip.hipDeviceSynchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    
    # Calculate metrics
    flops = 4 * batch * num_heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed * 1e9)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Time: {elapsed:.3f} ms")
    print(f"  TFLOPS: {tflops:.1f}")
    print(f"{'='*60}")
    
    return elapsed, tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FP8 Flash Attention Kernel")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark(args.batch, args.seq_len, args.heads, args.head_dim, args.warmup, args.iters)
    else:
        test_fp8_kernel(args.batch, args.seq_len, args.heads, args.head_dim)

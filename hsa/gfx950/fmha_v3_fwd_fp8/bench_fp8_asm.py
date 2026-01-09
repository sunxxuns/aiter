#!/usr/bin/env python3
"""
Direct FP8 ASM Kernel Benchmark

Benchmarks the FP8 assembly kernel directly using HIP.
Compares with BF16 baseline from AITER.
"""

import os
import sys
import time
import ctypes
import struct

import torch

# HIP library
try:
    libhip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
except:
    print("ERROR: Cannot load libamdhip64.so")
    sys.exit(1)

# Define HIP types
hipError_t = ctypes.c_int
hipModule_t = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p
hipStream_t = ctypes.c_void_p

# HIP API
libhip.hipModuleLoad.argtypes = [ctypes.POINTER(hipModule_t), ctypes.c_char_p]
libhip.hipModuleLoad.restype = hipError_t

libhip.hipModuleGetFunction.argtypes = [ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.c_char_p]
libhip.hipModuleGetFunction.restype = hipError_t

libhip.hipModuleLaunchKernel.argtypes = [
    hipFunction_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, hipStream_t, ctypes.c_void_p, ctypes.c_void_p
]
libhip.hipModuleLaunchKernel.restype = hipError_t

libhip.hipDeviceSynchronize.argtypes = []
libhip.hipDeviceSynchronize.restype = hipError_t

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP8_CO = os.path.join(SCRIPT_DIR, "fwd_hd128_fp8.co")

def get_kernel_name(co_path):
    import subprocess
    result = subprocess.run(
        ['/opt/rocm/llvm/bin/llvm-readelf', '--symbols', co_path],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if '.kd' in line and 'OBJECT' in line:
            parts = line.split()
            for p in parts:
                if '.kd' in p:
                    return p.replace('.kd', '')
    return None


class ASMKernel:
    def __init__(self, co_path, kernel_name=None):
        self.co_path = co_path
        self.module = hipModule_t()
        self.function = hipFunction_t()
        
        err = libhip.hipModuleLoad(ctypes.byref(self.module), co_path.encode())
        if err != 0:
            raise RuntimeError(f"hipModuleLoad failed: {err}")
        
        if kernel_name is None:
            kernel_name = get_kernel_name(co_path)
        self.kernel_name = kernel_name
        
        err = libhip.hipModuleGetFunction(
            ctypes.byref(self.function), 
            self.module, 
            kernel_name.encode()
        )
        if err != 0:
            raise RuntimeError(f"hipModuleGetFunction failed for {kernel_name}: {err}")
    
    def launch(self, args_buffer, grid, block, shared_mem=0, stream=None):
        args_ptr = ctypes.cast(args_buffer, ctypes.c_void_p)
        args_size = ctypes.c_size_t(len(args_buffer))
        
        HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
        HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
        HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)
        
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            args_ptr,
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(args_size), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END
        )
        
        err = libhip.hipModuleLaunchKernel(
            self.function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, stream,
            None, ctypes.cast(extra, ctypes.c_void_p)
        )
        if err != 0:
            raise RuntimeError(f"hipModuleLaunchKernel failed: {err}")


def pack_fp8_args(q, k, v, out, lse, softmax_scale, seqlen_q, seqlen_k, q_scale, k_scale, v_scale):
    """Pack arguments for FP8 kernel based on its argument layout."""
    # Based on the kernel metadata:
    # ptr_R @0, ptr_Q @16, ptr_K @32, ptr_V @48, ptr_LSE @64
    # softmax_scale @80, seqlen_q @88, seqlen_k @96
    # q_scale @512, k_scale @516, v_scale @520
    
    # Create a 528-byte buffer (kernarg_segment_size from metadata)
    args = bytearray(528)
    
    # Pack pointers and values
    struct.pack_into('Q', args, 0, out.data_ptr())      # ptr_R
    struct.pack_into('Q', args, 16, q.data_ptr())       # ptr_Q
    struct.pack_into('Q', args, 32, k.data_ptr())       # ptr_K
    struct.pack_into('Q', args, 48, v.data_ptr())       # ptr_V
    struct.pack_into('Q', args, 64, lse.data_ptr())     # ptr_LSE
    struct.pack_into('f', args, 80, softmax_scale)      # softmax_scale
    struct.pack_into('I', args, 88, seqlen_q)           # seqlen_q
    struct.pack_into('I', args, 96, seqlen_k)           # seqlen_k
    struct.pack_into('f', args, 512, q_scale)           # q_scale
    struct.pack_into('f', args, 516, k_scale)           # k_scale
    struct.pack_into('f', args, 520, v_scale)           # v_scale
    
    return bytes(args)


def benchmark_fp8_asm(batch, seq_len, num_heads, head_dim, warmup=10, iters=50):
    """Benchmark FP8 ASM kernel directly."""
    
    kernel = ASMKernel(FP8_CO)
    
    # Create FP8 tensors (randn doesn't support fp8 directly, so convert from bf16)
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    out = torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    lse = torch.zeros(batch, num_heads, seq_len, dtype=torch.float32, device='cuda')
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    q_scale = k_scale = v_scale = 1.0
    
    args = pack_fp8_args(q, k, v, out, lse, softmax_scale, seq_len, seq_len, q_scale, k_scale, v_scale)
    
    # Grid/block - match BLOCK_M=64 in kernel
    ts_qo = 64  # BLOCK_M from assembly
    grid = ((seq_len + ts_qo - 1) // ts_qo, num_heads, batch)
    block = (256, 1, 1)  # 4 warps * 64
    shared_mem = 32768  # 32KB from kernel metadata
    
    # Warmup
    for _ in range(warmup):
        kernel.launch(args, grid, block, shared_mem)
    libhip.hipDeviceSynchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        kernel.launch(args, grid, block, shared_mem)
    libhip.hipDeviceSynchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    
    # FLOPs calculation (same as attention)
    flops = 4 * batch * num_heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed * 1e9)
    
    return elapsed, tflops


def benchmark_bf16_aiter(batch, seq_len, num_heads, head_dim, warmup=10, iters=50):
    """Benchmark BF16 via AITER's fmha_v3_fwd."""
    from aiter.ops.mha import fmha_v3_fwd
    
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    
    softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Warmup
    for _ in range(warmup):
        _ = fmha_v3_fwd(q, k, v, 0.0, softmax_scale, False, -1, -1, True, False, 1)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = fmha_v3_fwd(q, k, v, 0.0, softmax_scale, False, -1, -1, True, False, 1)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    
    flops = 4 * batch * num_heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed * 1e9)
    
    return elapsed, tflops


def main():
    print("=" * 80)
    print("FP8 vs BF16 ASM Flash Attention Benchmark")
    print("=" * 80)
    
    batch, num_heads, head_dim = 1, 40, 128
    seq_lens = [4096, 8192, 16384, 32130]
    
    print(f"\nConfiguration: B={batch}, H={num_heads}, D={head_dim}")
    print("-" * 80)
    print(f"{'SeqLen':>10} {'BF16 ASM (ms)':>15} {'BF16 TF/s':>12} {'FP8 ASM (ms)':>15} {'FP8 TF/s':>12}")
    print("-" * 80)
    
    for seq_len in seq_lens:
        # BF16 baseline via AITER
        try:
            bf16_ms, bf16_tf = benchmark_bf16_aiter(batch, seq_len, num_heads, head_dim)
            bf16_str = f"{bf16_ms:.3f}"
            bf16_tf_str = f"{bf16_tf:.1f}"
        except Exception as e:
            bf16_str = "ERROR"
            bf16_tf_str = str(e)[:20]
        
        # FP8 ASM kernel
        try:
            fp8_ms, fp8_tf = benchmark_fp8_asm(batch, seq_len, num_heads, head_dim)
            fp8_str = f"{fp8_ms:.3f}"
            fp8_tf_str = f"{fp8_tf:.1f}"
        except Exception as e:
            fp8_str = "ERROR"
            fp8_tf_str = str(e)[:20]
        
        print(f"{seq_len:>10} {bf16_str:>15} {bf16_tf_str:>12} {fp8_str:>15} {fp8_tf_str:>12}")
    
    print("-" * 80)
    print()
    print("NOTE: FP8 kernel has 48 MFMAs (vs 176 in BF16)")
    print("      FP8 uses 3.3x less LDS (48KB vs 160KB)")
    print("      Target FP8 performance: >1300 TF/s (30% above BF16 baseline)")
    print("=" * 80)


if __name__ == "__main__":
    main()

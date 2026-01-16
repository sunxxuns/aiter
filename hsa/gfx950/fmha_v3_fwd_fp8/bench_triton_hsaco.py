#!/usr/bin/env python3
"""
Benchmark Triton's compiled FP8 flash attention HSACO directly.

Arguments from triton_fp8_fmha.s metadata (120 bytes total):
- offset 0:   Q ptr (8 bytes)
- offset 8:   K ptr (8 bytes)  
- offset 16:  V ptr (8 bytes)
- offset 24:  sm_scale (4 bytes, float)
- offset 32:  M ptr (8 bytes) - log-sum-exp output
- offset 40:  Out ptr (8 bytes)
- offset 48:  stride_qz (4 bytes)
- offset 52:  stride_qh (4 bytes)
- offset 56:  stride_qm (4 bytes)
- offset 60:  stride_qk (4 bytes)
- offset 64:  stride_kz (4 bytes)
- offset 68:  stride_kh (4 bytes)
- offset 72:  stride_kn (4 bytes)
- offset 76:  stride_kk (4 bytes)
- offset 80:  stride_vz (4 bytes)
- offset 84:  stride_vh (4 bytes)
- offset 88:  stride_vk (4 bytes)
- offset 92:  stride_vn (4 bytes)
- offset 96:  stride_oz (4 bytes)
- offset 100: stride_oh (4 bytes)
- offset 104: stride_om (4 bytes) - metadata says ptr but likely stride
- offset 108: stride_on (4 bytes)
- offset 112: Z (4 bytes)
- offset 116: H (4 bytes)
- Note: N_CTX might be a constexpr baked into the kernel
"""

import os
import ctypes
import struct
import math
import subprocess

os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def compile_triton_asm():
    """Compile the Triton assembly to HSACO."""
    src = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s"
    obj = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.co"
    
    cmd = f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o {obj} {src}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compile error: {result.stderr}")
        return False
    print(f"Compiled {obj}")
    return True

def load_kernel():
    """Load the Triton kernel."""
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    path = b'/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.co'
    err = hip.hipModuleLoad(ctypes.byref(module), path)
    if err != 0:
        print(f"hipModuleLoad failed: {err}")
        return None, None
    
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b'_attn_fwd')
    if err != 0:
        print(f"hipModuleGetFunction failed: {err}")
        return None, None
        
    print("Kernel loaded successfully")
    return module, func

def benchmark(func, batch=1, heads=40, seq_len=32130, head_dim=128, warmup=5, iters=20):
    """Benchmark the Triton FP8 attention kernel."""
    
    print(f"\n{'='*70}")
    print(f"Triton FP8 Flash Attention Benchmark")
    print(f"B={batch}, H={heads}, S={seq_len}, D={head_dim}")
    print(f"{'='*70}")
    
    # Create tensors - shape is (batch, heads, seq_len, head_dim)
    Q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    K = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    V = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    
    # Output in float32
    Out = torch.zeros(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    # M tensor for log-sum-exp (batch, heads, seq_len)
    M = torch.zeros(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Grid: (num_q_blocks, batch * heads, 1)
    BLOCK_M = 128
    num_q_blocks = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_x = num_q_blocks
    grid_y = batch * heads
    
    print(f"Grid: ({grid_x}, {grid_y}, 1), Block: (256, 1, 1)")
    print(f"Total blocks: {grid_x * grid_y}")
    
    # Build kernel arguments (120 bytes)
    # Using struct to pack properly
    kernarg = bytearray(120)
    
    # Pointers (8 bytes each)
    struct.pack_into('Q', kernarg, 0, Q.data_ptr())      # Q
    struct.pack_into('Q', kernarg, 8, K.data_ptr())      # K
    struct.pack_into('Q', kernarg, 16, V.data_ptr())     # V
    struct.pack_into('f', kernarg, 24, sm_scale)         # sm_scale
    struct.pack_into('Q', kernarg, 32, M.data_ptr())     # M
    struct.pack_into('Q', kernarg, 40, Out.data_ptr())   # Out
    
    # Q strides (in elements, not bytes)
    struct.pack_into('i', kernarg, 48, Q.stride(0))      # stride_qz
    struct.pack_into('i', kernarg, 52, Q.stride(1))      # stride_qh
    struct.pack_into('i', kernarg, 56, Q.stride(2))      # stride_qm
    struct.pack_into('i', kernarg, 60, Q.stride(3))      # stride_qk
    
    # K strides
    struct.pack_into('i', kernarg, 64, K.stride(0))      # stride_kz
    struct.pack_into('i', kernarg, 68, K.stride(1))      # stride_kh
    struct.pack_into('i', kernarg, 72, K.stride(2))      # stride_kn
    struct.pack_into('i', kernarg, 76, K.stride(3))      # stride_kk
    
    # V strides
    struct.pack_into('i', kernarg, 80, V.stride(0))      # stride_vz
    struct.pack_into('i', kernarg, 84, V.stride(1))      # stride_vh
    struct.pack_into('i', kernarg, 88, V.stride(2))      # stride_vk
    struct.pack_into('i', kernarg, 92, V.stride(3))      # stride_vn
    
    # O strides
    struct.pack_into('i', kernarg, 96, Out.stride(0))    # stride_oz
    struct.pack_into('i', kernarg, 100, Out.stride(1))   # stride_oh
    struct.pack_into('i', kernarg, 104, Out.stride(2))   # stride_om
    struct.pack_into('i', kernarg, 108, Out.stride(3))   # stride_on
    
    # Z, H, N_CTX
    struct.pack_into('i', kernarg, 112, batch)           # Z
    struct.pack_into('i', kernarg, 116, heads)           # H
    # N_CTX might be baked in as constexpr - let's try without it first
    
    print(f"\nStrides:")
    print(f"  Q: {Q.stride()}")
    print(f"  K: {K.stride()}")
    print(f"  V: {V.stride()}")
    print(f"  Out: {Out.stride()}")
    
    # Allocate kernarg on device
    kernarg_ptr = ctypes.c_void_p()
    hip.hipMalloc(ctypes.byref(kernarg_ptr), 256)  # Extra space
    hip.hipMemcpy(kernarg_ptr, (ctypes.c_char * len(kernarg)).from_buffer(kernarg), 
                  len(kernarg), 1)  # hipMemcpyHostToDevice
    
    # LDS size - Triton kernel metadata says 0, but let's try 16KB
    lds_size = 16384
    
    print(f"\nLaunching kernel...")
    
    # Try single launch first
    try:
        err = hip.hipModuleLaunchKernel(
            func,
            grid_x, grid_y, 1,  # grid
            256, 1, 1,          # block (256 threads = 4 warps)
            lds_size,           # shared memory
            None,               # stream
            kernarg_ptr,        # kernarg
            None                # extra
        )
        hip.hipDeviceSynchronize()
        
        if err != 0:
            print(f"Launch failed with error: {err}")
            return
            
        print("Launch successful!")
        
        # Check output
        print(f"\nOutput stats:")
        print(f"  Range: [{Out.min().item():.4f}, {Out.max().item():.4f}]")
        print(f"  Mean: {Out.mean().item():.4f}")
        print(f"  Has NaN: {torch.isnan(Out).any().item()}")
        print(f"  Has Inf: {torch.isinf(Out).any().item()}")
        
        # Warmup
        print(f"\nWarmup ({warmup} iters)...")
        for _ in range(warmup):
            hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, lds_size, None, kernarg_ptr, None)
        hip.hipDeviceSynchronize()
        
        # Benchmark
        print(f"Benchmark ({iters} iters)...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, lds_size, None, kernarg_ptr, None)
        end.record()
        end.synchronize()
        
        avg_ms = start.elapsed_time(end) / iters
        
        # FLOPs for full attention
        flops = 4 * batch * heads * seq_len * seq_len * head_dim
        tflops = flops / (avg_ms * 1e-3) / 1e12
        
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"Average time: {avg_ms:.3f} ms")
        print(f"Performance: {tflops:.1f} TF/s")
        
    except Exception as e:
        print(f"Error: {e}")
    
    hip.hipFree(kernarg_ptr)


if __name__ == "__main__":
    if not compile_triton_asm():
        exit(1)
    
    module, func = load_kernel()
    if func is None:
        exit(1)
    
    # Small test first
    benchmark(func, batch=1, heads=1, seq_len=1024, head_dim=128, warmup=2, iters=5)
    
    # Target benchmark
    # benchmark(func, batch=1, heads=40, seq_len=32130, head_dim=128)
    
    hip.hipModuleUnload(module)

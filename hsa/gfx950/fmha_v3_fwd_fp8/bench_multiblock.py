#!/usr/bin/env python3
"""Benchmark FP8 QK multi-block kernel at scale."""

import torch
import ctypes
import subprocess
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')  # Initialize CUDA first

def compile_kernel():
    """Compile the multi-block kernel."""
    src = "fwd_fp8_qk_multiblock.s"
    obj = "fwd_fp8_qk_multiblock.co"
    
    cmd = f"cd /sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8 && /opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o {obj} {src}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compile error: {result.stderr}")
        return False
    print(f"Compiled {obj}")
    return True

def load_kernel():
    """Load the multi-block kernel."""
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), b'/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_multiblock.co')
    hip.hipModuleGetFunction(ctypes.byref(func), module, b'_ZN5aiter19fwd_fp8_qk_multiblockE')
    
    return module, func

def benchmark_multiblock(func, seq_len=32128, num_heads=40, head_dim=128, warmup=5, iters=20):
    """Benchmark multi-block kernel."""
    
    num_q_tiles = (seq_len + 31) // 32
    
    print(f"\n=== Multi-block FP8 QK Benchmark ===")
    print(f"seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"num_q_tiles={num_q_tiles}")
    
    # Allocate tensors
    Q = torch.randn(num_heads, seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    K = torch.randn(num_heads, seq_len, head_dim, device='cuda').to(torch.float8_e4m3fn)
    
    # Output: each block writes 256*64 bytes = 16KB
    O = torch.zeros(num_heads, num_q_tiles, 256, 16, dtype=torch.float32, device='cuda')
    
    # Strides (in bytes)
    Q_stride_seq = 32 * head_dim      # 32 rows * 128 bytes = 4096
    Q_stride_head = seq_len * head_dim
    K_stride_head = seq_len * head_dim
    O_stride_seq = 256 * 64           # 16384 bytes per Q tile
    O_stride_head = num_q_tiles * O_stride_seq
    
    # Kernel arguments (working pattern)
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(Q_stride_seq),
        ctypes.c_uint32(Q_stride_head),
        ctypes.c_uint32(K_stride_head),
        ctypes.c_uint32(O_stride_seq),
        ctypes.c_uint32(O_stride_head),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Grid: (num_q_tiles, num_heads, 1)
    grid_x = num_q_tiles
    grid_y = num_heads
    
    print(f"Grid: ({grid_x}, {grid_y}, 1), Block: (256, 1, 1)")
    print(f"Total blocks: {grid_x * grid_y}")
    
    # Warmup
    print(f"Warmup ({warmup} iters)...")
    for _ in range(warmup):
        hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Timed iterations
    print(f"Benchmark ({iters} iters)...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    end.synchronize()
    
    avg_ms = start.elapsed_time(end) / iters
    
    # Calculate TF/s
    # QK FLOPs = 2 * num_heads * num_q_tiles * 32 * seq_len * head_dim
    flops = 2 * num_heads * num_q_tiles * 32 * seq_len * head_dim
    tflops = flops / (avg_ms * 1e9)
    
    print(f"\n=== Results ===")
    print(f"Average time: {avg_ms:.3f} ms")
    print(f"Total FLOPs: {flops:.2e}")
    print(f"Performance: {tflops:.1f} TF/s")
    
    return tflops

def test_correctness(func):
    """Test numerical correctness with small inputs."""
    print("\n=== Correctness Test ===")
    
    seq_len = 64
    num_heads = 1
    head_dim = 128
    num_q_tiles = (seq_len + 31) // 32
    
    # Simple test: all ones
    Q = torch.ones(num_heads, seq_len, head_dim, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(num_heads, seq_len, head_dim, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(num_heads, num_q_tiles, 256, 16, dtype=torch.float32, device='cuda')
    
    Q_stride_seq = 32 * head_dim
    Q_stride_head = seq_len * head_dim
    K_stride_head = seq_len * head_dim
    O_stride_seq = 256 * 64
    O_stride_head = num_q_tiles * O_stride_seq
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(Q_stride_seq),
        ctypes.c_uint32(Q_stride_head),
        ctypes.c_uint32(K_stride_head),
        ctypes.c_uint32(O_stride_seq),
        ctypes.c_uint32(O_stride_head),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Launch single block
    hip.hipModuleLaunchKernel(func, num_q_tiles, num_heads, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    print(f"Output shape: {O.shape}")
    print(f"Output range: [{O.min().item():.2f}, {O.max().item():.2f}]")
    print(f"Output mean: {O.mean().item():.2f}")
    print(f"First 16 values: {O[0, 0, 0, :16].tolist()}")
    
    has_nan = torch.isnan(O).any().item()
    has_inf = torch.isinf(O).any().item()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
    
    return not has_nan and not has_inf

if __name__ == "__main__":
    if not compile_kernel():
        exit(1)
    
    module, func = load_kernel()
    
    # Test correctness
    if not test_correctness(func):
        print("Correctness test failed!")
        exit(1)
    
    # Benchmarks
    print("\n" + "="*60)
    print("BENCHMARKS")
    print("="*60)
    
    # Small scale
    benchmark_multiblock(func, seq_len=1024, num_heads=1, head_dim=128)
    
    # Medium scale
    benchmark_multiblock(func, seq_len=4096, num_heads=8, head_dim=128)
    
    # Target scale
    benchmark_multiblock(func, seq_len=32128, num_heads=40, head_dim=128)
    
    hip.hipModuleUnload(module)

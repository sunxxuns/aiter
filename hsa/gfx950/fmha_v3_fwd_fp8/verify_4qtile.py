#!/usr/bin/env python3
"""
Verify TF/s for fwd_fp8_qk_4qtile_v2.s with random input.
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def get_func_name(co_file):
    """Get function name from .co file."""
    import subprocess
    result = subprocess.run(
        f"llvm-readelf -s {co_file} | grep 'FUNC.*kd' | head -1",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        # Extract function name
        parts = result.stdout.strip().split()
        for p in parts:
            if p.startswith('_ZN'):
                return p.replace('.kd', '')
    return None

def benchmark_kernel(co_file, func_name, seq_len, num_heads, head_dim, q_rows_per_block):
    """Benchmark kernel and calculate correct TF/s."""
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    if not os.path.exists(co_file):
        print(f"Kernel not found: {co_file}")
        return None
    
    err = hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    if err != 0:
        print(f"hipModuleLoad error: {err}")
        return None
    
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    if err != 0:
        print(f"hipModuleGetFunction error: {err} for {func_name}")
        hip.hipModuleUnload(module)
        return None
    
    # Random input
    torch.manual_seed(42)
    Q = (torch.randn(seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    K = (torch.randn(seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    O = torch.zeros(seq_len * seq_len, dtype=torch.float32, device='cuda')
    
    # Number of blocks
    num_q_tiles = (seq_len + q_rows_per_block - 1) // q_rows_per_block
    num_blocks = num_q_tiles * num_heads
    
    # Kernel args
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # LDS size based on kernel
    lds_size = 32768 if q_rows_per_block > 32 else 16384
    
    # Warmup
    for _ in range(3):
        err = hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, lds_size, None, args_ptrs, None)
        if err != 0:
            print(f"Launch error: {err}")
            hip.hipModuleUnload(module)
            return None
    hip.hipDeviceSynchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 10
    start.record()
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, lds_size, None, args_ptrs, None)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    
    hip.hipModuleUnload(module)
    
    # FLOPS calculation
    # Per block: q_rows_per_block * seq_len * head_dim * 2
    flops_per_block = 2 * q_rows_per_block * seq_len * head_dim
    total_flops = flops_per_block * num_blocks
    
    tflops = total_flops / (time_ms / 1000) / 1e12
    
    return {
        'time_ms': time_ms,
        'tflops': tflops,
        'num_blocks': num_blocks,
        'total_flops': total_flops,
    }

def main():
    print("=" * 70)
    print("TF/s VERIFICATION FOR 4QTILE_V2 KERNEL")
    print("=" * 70)
    
    # Test configurations
    kernels = [
        ("fwd_fp8_qk_preload", "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co", 
         "_ZN5aiter17fwd_fp8_qk_preloadE", 32),
        ("fwd_fp8_qk_4qtile_v2", "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_4qtile_v2.co",
         "_ZN5aiter19fwd_fp8_qk_4qtile_v2E", 128),
    ]
    
    seq_len = 32128
    num_heads = 40
    head_dim = 128
    
    print(f"\nTest config: seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"{'Kernel':<25} | {'Q rows/blk':>10} | {'blocks':>8} | {'time (ms)':>10} | {'TF/s':>10}")
    print("-" * 80)
    
    for name, co_file, func_name, q_rows in kernels:
        result = benchmark_kernel(co_file, func_name, seq_len, num_heads, head_dim, q_rows)
        if result:
            print(f"{name:<25} | {q_rows:>10} | {result['num_blocks']:>8} | {result['time_ms']:>10.3f} | {result['tflops']:>10.1f}")
        else:
            print(f"{name:<25} | FAILED")
    
    # Numerical verification for 4qtile_v2
    print("\n" + "=" * 70)
    print("NUMERICAL VERIFICATION FOR 4QTILE_V2")
    print("=" * 70)
    
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_4qtile_v2.co"
    func_name = b"_ZN5aiter19fwd_fp8_qk_4qtile_v2E"
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    err = hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    if err != 0:
        print(f"hipModuleLoad error: {err}")
        return
    
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    if err != 0:
        print(f"hipModuleGetFunction error: {err}")
        hip.hipModuleUnload(module)
        return
    
    seq_len = 128  # Small for verification
    head_dim = 128
    
    # Random input
    torch.manual_seed(42)
    Q = (torch.randn(seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    K = (torch.randn(seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    O = torch.zeros(seq_len * seq_len, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    err = hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 32768, None, args_ptrs, None)
    if err != 0:
        print(f"Launch error: {err}")
        hip.hipModuleUnload(module)
        return
    hip.hipDeviceSynchronize()
    
    # PyTorch reference
    Q_f32 = Q[:128, :].to(torch.float32)
    K_f32 = K.to(torch.float32)
    
    # Full QK for first 128x128 block
    num_k_tiles = (seq_len + 31) // 32
    ref = torch.zeros(128, 128, device='cuda')
    for kt in range(num_k_tiles):
        k_start = kt * 32
        k_end = min(k_start + 32, seq_len)
        K_tile = K_f32[k_start:k_end, :]
        if K_tile.shape[0] < 32:
            K_tile = torch.nn.functional.pad(K_tile, (0, 0, 0, 32 - K_tile.shape[0]))
        # Each Q-tile (32 rows) multiplies with this K-tile
        for qt in range(4):
            q_start = qt * 32
            q_end = q_start + 32
            ref[q_start:q_end, k_start:min(k_end, k_start+32)] += Q_f32[q_start:q_end, :] @ K_tile.T
    
    hip.hipModuleUnload(module)
    
    print(f"\nTest seq_len={seq_len}")
    print(f"Reference shape: {ref.shape}")
    print(f"Reference [0,0]: {ref[0,0].item():.4f}")
    print(f"Reference mean: {ref.mean().item():.4f}")
    print(f"Reference range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")
    
    # Kernel outputs 128x128 results
    kernel_out = O[:128*128].reshape(128, 128)
    print(f"\nKernel [0,0]: {kernel_out[0,0].item():.4f}")
    print(f"Kernel mean: {kernel_out.mean().item():.4f}")
    print(f"Kernel range: [{kernel_out.min().item():.4f}, {kernel_out.max().item():.4f}]")
    
    # Compare sorted values
    ref_sorted = torch.sort(ref.flatten())[0]
    out_sorted = torch.sort(kernel_out.flatten())[0]
    diff = (ref_sorted - out_sorted).abs().mean().item()
    print(f"\nSorted diff: {diff:.6f}")
    print(f"Match: {diff < 0.1}")

if __name__ == "__main__":
    main()

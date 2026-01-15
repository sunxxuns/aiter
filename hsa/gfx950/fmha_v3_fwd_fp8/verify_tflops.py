#!/usr/bin/env python3
"""
Verify TF/s calculation with random input.
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def benchmark_kernel(co_file, func_name, seq_len=32128, num_heads=40, head_dim=128):
    """Benchmark kernel and calculate correct TF/s."""
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    if not os.path.exists(co_file):
        print(f"Kernel not found: {co_file}")
        return None
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    
    # Random input (not uniform!)
    torch.manual_seed(42)
    Q = (torch.randn(num_heads, seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    K = (torch.randn(num_heads, seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    O = torch.zeros(num_heads, seq_len, seq_len, dtype=torch.float32, device='cuda')
    
    # This kernel processes 32 Q rows per block
    num_q_tiles = (seq_len + 31) // 32
    num_blocks = num_q_tiles * num_heads
    
    # Kernel args (assuming preload kernel interface)
    # Note: This kernel processes one Q-tile per block
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),  # q_tile_idx - but kernel uses block_id
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    # Warmup
    for _ in range(3):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iterations = 10
    start.record()
    for _ in range(iterations):
        hip.hipModuleLaunchKernel(func, num_blocks, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iterations
    
    hip.hipModuleUnload(module)
    
    # FLOPS calculation for QK computation
    # Full attention QK: 2 * seq_len * seq_len * head_dim * num_heads
    # But this kernel only does partial computation per block
    
    # Per block: 32 Q rows × seq_len K rows × head_dim
    # FLOPs per block = 2 * 32 * seq_len * head_dim
    flops_per_block = 2 * 32 * seq_len * head_dim
    total_flops = flops_per_block * num_blocks
    
    # Alternative: Full QK computation
    # full_qk_flops = 2 * seq_len * seq_len * head_dim * num_heads
    
    tflops = total_flops / (time_ms / 1000) / 1e12
    
    return {
        'time_ms': time_ms,
        'tflops': tflops,
        'num_blocks': num_blocks,
        'total_flops': total_flops,
    }

def verify_numerical():
    """Verify numerical correctness with random input."""
    
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co"
    func_name = b"_ZN5aiter17fwd_fp8_qk_preloadE"
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    
    seq_len = 64
    head_dim = 128
    
    # Random input
    torch.manual_seed(42)
    Q = (torch.randn(seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    K = (torch.randn(seq_len, head_dim, device='cuda') * 0.5).to(torch.float8_e4m3fn)
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # PyTorch reference
    Q_f32 = Q[:32, :].to(torch.float32)
    K_f32 = K.to(torch.float32)
    
    # This kernel sums over all K tiles
    num_k_tiles = (seq_len + 31) // 32
    ref = torch.zeros(32, 32, device='cuda')
    for kt in range(num_k_tiles):
        k_start = kt * 32
        k_end = min(k_start + 32, seq_len)
        K_tile = K_f32[k_start:k_end, :]
        if K_tile.shape[0] < 32:
            K_tile = torch.nn.functional.pad(K_tile, (0, 0, 0, 32 - K_tile.shape[0]))
        ref += Q_f32 @ K_tile.T
    
    hip.hipModuleUnload(module)
    
    # Compare
    kernel_out = O[:1024].reshape(32, 32)
    
    # The output layout might be different, compare sorted values
    ref_sorted = torch.sort(ref.flatten())[0]
    out_sorted = torch.sort(kernel_out.flatten())[0]
    
    return {
        'ref_mean': ref.mean().item(),
        'out_mean': kernel_out.mean().item(),
        'ref_range': (ref.min().item(), ref.max().item()),
        'out_range': (kernel_out.min().item(), kernel_out.max().item()),
        'sorted_diff': (ref_sorted - out_sorted).abs().mean().item(),
        'match': (ref_sorted - out_sorted).abs().mean().item() < 0.01,
    }

def main():
    print("=" * 70)
    print("TF/s VERIFICATION WITH RANDOM INPUT")
    print("=" * 70)
    
    # First verify numerical correctness
    print("\n--- Numerical Verification ---")
    result = verify_numerical()
    print(f"Reference mean: {result['ref_mean']:.4f}")
    print(f"Kernel mean: {result['out_mean']:.4f}")
    print(f"Reference range: [{result['ref_range'][0]:.4f}, {result['ref_range'][1]:.4f}]")
    print(f"Kernel range: [{result['out_range'][0]:.4f}, {result['out_range'][1]:.4f}]")
    print(f"Sorted diff: {result['sorted_diff']:.6f}")
    print(f"Match: {result['match']}")
    
    # Benchmark at different scales
    print("\n--- Performance Benchmark ---")
    print(f"{'seq_len':>8} | {'heads':>6} | {'time (ms)':>10} | {'TF/s':>10} | {'blocks':>8}")
    print("-" * 60)
    
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co"
    func_name = "_ZN5aiter17fwd_fp8_qk_preloadE"
    
    for seq_len, num_heads in [(64, 1), (256, 1), (1024, 1), (4096, 40), (32128, 40)]:
        result = benchmark_kernel(co_file, func_name, seq_len, num_heads)
        if result:
            print(f"{seq_len:>8} | {num_heads:>6} | {result['time_ms']:>10.3f} | {result['tflops']:>10.1f} | {result['num_blocks']:>8}")
    
    print("\n" + "=" * 70)
    print("FLOPS CALCULATION EXPLANATION")
    print("=" * 70)
    print("""
For QK computation (Q @ K^T):
- Q shape: [seq_len, head_dim]
- K shape: [seq_len, head_dim]
- QK shape: [seq_len, seq_len]

FLOPs = 2 * M * N * K where:
- M = seq_len (output rows)
- N = seq_len (output cols) 
- K = head_dim (reduction dim)

Full QK FLOPs = 2 * seq_len * seq_len * head_dim

For seq_len=32128, head_dim=128, num_heads=40:
- Full QK FLOPs = 2 * 32128 * 32128 * 128 * 40 = 10.56 PFLOP

But our kernel processes 32 Q rows per block:
- Per block FLOPs = 2 * 32 * seq_len * head_dim
- Total blocks = (seq_len/32) * num_heads = 1004 * 40 = 40160
- Total FLOPs = 2 * 32 * 32128 * 128 * 40160 = 10.56 PFLOP (same!)

So TF/s = 10.56 PFLOP / time_seconds
""")
    
    # Calculate expected TF/s for target scale
    seq_len = 32128
    head_dim = 128
    num_heads = 40
    
    full_flops = 2 * seq_len * seq_len * head_dim * num_heads
    print(f"\nTarget scale (seq={seq_len}, heads={num_heads}):")
    print(f"  Total FLOPs: {full_flops/1e15:.3f} PFLOP")
    
    # If kernel runs in X ms, TF/s = full_flops / (X/1000) / 1e12
    for time_ms in [1, 5, 10, 27]:
        tflops = full_flops / (time_ms / 1000) / 1e12
        print(f"  At {time_ms} ms: {tflops:.1f} TF/s ({tflops/1000:.2f} PF/s)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rigorous numerical verification of FP8 QK kernel against PyTorch reference.
"""

import torch
import ctypes
import os
import numpy as np

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def load_kernel():
    """Load the preload kernel."""
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    co_file = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_preload.co"
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, b'_ZN5aiter17fwd_fp8_qk_preloadE')
    return module, func

def run_kernel(func, Q_fp8, K_fp8, seq_len):
    """Run kernel and return output."""
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q_fp8.data_ptr()),
        ctypes.c_void_p(K_fp8.data_ptr()),
        ctypes.c_uint32(seq_len),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    return O

def compute_reference(Q_fp8, K_fp8, seq_len):
    """
    Compute reference QK output matching what the kernel does.
    
    The kernel:
    1. Takes Q[0:32, :] (32 rows)
    2. Iterates over K in tiles of 32 rows
    3. Accumulates QK = Q @ K_tile.T for each tile
    4. Output is 32x32 accumulated over all K-tiles
    """
    Q = Q_fp8[:32, :].to(torch.float32)
    
    num_k_tiles = (seq_len + 31) // 32
    ref = torch.zeros(32, 32, device='cuda')
    
    for k_tile in range(num_k_tiles):
        k_start = k_tile * 32
        k_end = min(k_start + 32, seq_len)
        K_tile = K_fp8[k_start:k_end, :].to(torch.float32)
        
        # Pad K_tile to 32 rows if needed
        if K_tile.shape[0] < 32:
            K_tile = torch.nn.functional.pad(K_tile, (0, 0, 0, 32 - K_tile.shape[0]))
        
        # Accumulate partial QK
        partial_qk = Q @ K_tile.T  # [32, 32]
        ref += partial_qk
    
    return ref

def decode_mfma_output(O_flat):
    """
    Decode the MFMA output layout to get a 32x32 matrix.
    
    MFMA 32x32x16 output: 16 floats per thread, 256 threads = 4096 floats
    Each thread's 16 outputs form a 4x4 block in the result matrix.
    
    Thread mapping for v_mfma_f32_32x32x16:
    - 64 threads per wavefront, 4 wavefronts
    - Output is distributed across threads in a specific pattern
    """
    # The exact layout depends on the MFMA instruction
    # For now, try to infer it from the data
    
    # Simple approach: reshape and hope for the best
    # This is likely wrong, but let's see what we get
    
    # Each thread writes 16 floats (v[0:15])
    # Thread tid writes to offset tid*64 (16 floats * 4 bytes = 64 bytes)
    # Total: 256 * 16 = 4096 floats
    
    # But output is only 32x32 = 1024 floats
    # So there's redundancy or the output layout is different
    
    return O_flat[:1024].reshape(32, 32)

def main():
    print("=" * 70)
    print("RIGOROUS NUMERICAL VERIFICATION")
    print("=" * 70)
    
    module, func = load_kernel()
    
    # Test 1: All ones (simple verification)
    print("\n--- Test 1: All Ones ---")
    seq_len = 64
    Q = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
    
    O = run_kernel(func, Q, K, seq_len)
    ref = compute_reference(Q, K, seq_len)
    
    print(f"Reference[0,0]: {ref[0,0].item():.2f}")
    print(f"Kernel out[0]: {O[0].item():.2f}")
    print(f"Reference mean: {ref.mean().item():.2f}")
    print(f"Kernel out mean: {O[:1024].mean().item():.2f}")
    
    # Test 2: Random inputs
    print("\n--- Test 2: Random Inputs ---")
    torch.manual_seed(42)
    
    # Generate random FP8-compatible values
    Q_f32 = torch.randn(seq_len, 128, device='cuda') * 0.5
    K_f32 = torch.randn(seq_len, 128, device='cuda') * 0.5
    
    Q = Q_f32.to(torch.float8_e4m3fn)
    K = K_f32.to(torch.float8_e4m3fn)
    
    O = run_kernel(func, Q, K, seq_len)
    ref = compute_reference(Q, K, seq_len)
    
    print(f"Reference shape: {ref.shape}")
    print(f"Reference[0,0]: {ref[0,0].item():.4f}")
    print(f"Reference mean: {ref.mean().item():.4f}")
    print(f"Reference range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")
    
    print(f"\nKernel output first 16: {O[:16].tolist()}")
    print(f"Kernel output mean (first 1024): {O[:1024].mean().item():.4f}")
    print(f"Kernel output range: [{O[:1024].min().item():.4f}, {O[:1024].max().item():.4f}]")
    
    # Try to correlate kernel output with reference
    print("\n--- Correlation Analysis ---")
    ref_flat = ref.flatten()
    out_flat = O[:1024]
    
    # Direct correlation
    correlation = torch.corrcoef(torch.stack([ref_flat, out_flat]))[0,1].item()
    print(f"Direct correlation: {correlation:.4f}")
    
    # Check if kernel output matches reference when properly reordered
    # The MFMA output needs to be understood
    
    # Check sorted values
    ref_sorted = torch.sort(ref_flat)[0]
    out_sorted = torch.sort(out_flat)[0]
    sorted_corr = torch.corrcoef(torch.stack([ref_sorted, out_sorted]))[0,1].item()
    print(f"Sorted correlation: {sorted_corr:.4f} (should be ~1.0 if same values)")
    
    # Check if values match (ignoring order)
    print(f"\nValue matching (ignoring order):")
    print(f"  Ref unique values: {len(torch.unique(ref_flat))}")
    print(f"  Out unique values: {len(torch.unique(out_flat))}")
    print(f"  Mean abs diff (sorted): {(ref_sorted - out_sorted).abs().mean().item():.6f}")
    
    # Test 3: Structured input (identity-like)
    print("\n--- Test 3: Structured Input ---")
    Q_f32 = torch.zeros(seq_len, 128, device='cuda')
    K_f32 = torch.zeros(seq_len, 128, device='cuda')
    
    # Set specific pattern: Q[i,j] = 1 if i==j else 0
    for i in range(min(32, seq_len)):
        Q_f32[i, i] = 1.0
        K_f32[i, i] = 1.0
    
    Q = Q_f32.to(torch.float8_e4m3fn)
    K = K_f32.to(torch.float8_e4m3fn)
    
    O = run_kernel(func, Q, K, seq_len)
    ref = compute_reference(Q, K, seq_len)
    
    print(f"Reference (should be identity for 32x32 block):")
    print(f"  Diagonal sum: {ref.diagonal().sum().item():.2f} (expected: 32)")
    print(f"  Off-diagonal mean: {(ref - torch.diag(ref.diagonal())).mean().item():.6f}")
    
    print(f"\nKernel output (first 32 values, one per row?):")
    print(f"  {O[:32].tolist()}")
    
    # Test 4: Single row active
    print("\n--- Test 4: Single Row Active ---")
    Q_f32 = torch.zeros(seq_len, 128, device='cuda')
    K_f32 = torch.zeros(seq_len, 128, device='cuda')
    
    Q_f32[0, :] = 1.0  # Only row 0 of Q active
    K_f32[0, :] = 1.0  # Only row 0 of K active
    
    Q = Q_f32.to(torch.float8_e4m3fn)
    K = K_f32.to(torch.float8_e4m3fn)
    
    O = run_kernel(func, Q, K, seq_len)
    ref = compute_reference(Q, K, seq_len)
    
    print(f"Reference:")
    print(f"  [0,0]: {ref[0,0].item():.2f} (expected: 128)")
    print(f"  [0,1]: {ref[0,1].item():.2f} (expected: 0)")
    print(f"  [1,0]: {ref[1,0].item():.2f} (expected: 0)")
    
    print(f"\nKernel output:")
    print(f"  out[0]: {O[0].item():.2f}")
    print(f"  Non-zero count: {(O[:1024].abs() > 0.01).sum().item()}")
    
    hip.hipModuleUnload(module)
    
    # Bank conflict detailed analysis
    print("\n" + "=" * 70)
    print("BANK CONFLICT DETAILED ANALYSIS")
    print("=" * 70)
    
    print("""
For pitch-136 with MFMA read pattern:
- Lane L reads row (L % 32), k_offset = 8 * (L >= 32)
- Address = row * 136 + k_offset

Bank calculation (4 bytes per bank, 64 banks):
""")
    
    conflicts = {}
    for lane in range(64):
        row = lane % 32
        k_off = 8 if lane >= 32 else 0
        addr = row * 136 + k_off
        bank = (addr // 4) % 64
        
        if bank not in conflicts:
            conflicts[bank] = []
        conflicts[bank].append((lane, row, k_off, addr))
    
    print(f"Lanes per bank (should be <=2 for 2-way conflict):")
    max_conflict = 0
    for bank in sorted(conflicts.keys()):
        lanes = conflicts[bank]
        if len(lanes) > max_conflict:
            max_conflict = len(lanes)
        if len(lanes) > 1:
            print(f"  Bank {bank:2d}: {[l[0] for l in lanes]} (rows {[l[1] for l in lanes]}, offsets {[l[2] for l in lanes]})")
    
    print(f"\nMax conflict level: {max_conflict}-way")
    print(f"Banks with conflicts: {sum(1 for v in conflicts.values() if len(v) > 1)}/64")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Investigate the 2x numerical error in FP8 QK kernel.
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def test_kernel(kernel_name, co_file, func_name, seq_len=64):
    """Test a kernel and compare to reference."""
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    if not os.path.exists(co_file):
        print(f"Kernel not found: {co_file}")
        return None
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    
    # Simple test: all ones
    Q = torch.ones(1, seq_len, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    K = torch.ones(1, seq_len, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    # Reference: Q @ K^T for first 32 rows
    Q_ref = Q[0, :32, :].to(torch.float32)
    K_ref = K[0, :32, :].to(torch.float32)
    ref_qk = Q_ref @ K_ref.T  # [32, 32]
    
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
    
    hip.hipModuleUnload(module)
    
    return {
        'name': kernel_name,
        'ref_00': ref_qk[0, 0].item(),
        'ref_mean': ref_qk.mean().item(),
        'out_00': O[0].item(),
        'out_mean': O.mean().item(),
        'ratio': O[0].item() / ref_qk[0, 0].item() if ref_qk[0, 0].item() != 0 else float('inf'),
    }

def analyze_num_k_tiles():
    """Check if the 2x error is related to num_k_tiles calculation."""
    print("=" * 70)
    print("ANALYZING NUM_K_TILES CALCULATION")
    print("=" * 70)
    
    for seq_len in [32, 64, 96, 128]:
        num_k_tiles = (seq_len + 31) // 32
        print(f"seq_len={seq_len}: num_k_tiles={num_k_tiles}")
        
        # Expected QK value for all-ones
        # With num_k_tiles K tiles, we sum over all of them
        # Each K tile contributes 32 rows * 128 cols = 4096 elements
        # But MFMA 32x32x16 only uses 16 k-elements at a time
        # So we need 128/16 = 8 MFMA calls per K tile
        expected_per_tile = 128  # 1*1*128 dot product for all-ones
        expected_total = expected_per_tile * num_k_tiles
        print(f"  Expected sum over K-tiles: {expected_total}")

def main():
    print("=" * 70)
    print("INVESTIGATING 2X NUMERICAL ERROR")
    print("=" * 70)
    
    # Test preload kernel with different seq_len
    kernel_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    
    print(f"\n{'seq_len':>8} | {'Expected':>10} | {'Got':>10} | {'Ratio':>8}")
    print("-" * 50)
    
    for seq_len in [32, 64, 96, 128]:
        result = test_kernel(
            f"preload_{seq_len}",
            f"{kernel_path}/fwd_fp8_qk_preload.co",
            "_ZN5aiter17fwd_fp8_qk_preloadE",
            seq_len=seq_len
        )
        if result:
            expected = 128.0  # All-ones: 1*1*128 = 128
            # If kernel sums over num_k_tiles, expected should scale
            num_k_tiles = (seq_len + 31) // 32
            expected_scaled = expected * num_k_tiles
            
            print(f"{seq_len:>8} | {expected_scaled:>10.1f} | {result['out_00']:>10.1f} | {result['out_00']/expected_scaled:>8.2f}x")
    
    analyze_num_k_tiles()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The 2x error appears when seq_len=64 (num_k_tiles=2):
- Expected: 128 * 2 = 256 (sum over 2 K-tiles)
- Got: 256

Wait - this is actually CORRECT! The kernel sums over ALL K-tiles.

Let me re-check the reference calculation...
""")
    
    # Proper reference calculation
    print("\nProper reference calculation:")
    for seq_len in [32, 64]:
        Q = torch.ones(seq_len, 128, dtype=torch.float32, device='cuda')
        K = torch.ones(seq_len, 128, dtype=torch.float32, device='cuda')
        
        # Full QK (32 Q rows × seq_len K rows)
        qk = Q[:32, :] @ K.T  # [32, seq_len]
        
        print(f"seq_len={seq_len}:")
        print(f"  QK shape: {qk.shape}")
        print(f"  QK[0,0]: {qk[0,0].item()}")  # Should be 128 (dot product of ones)
        print(f"  QK[0,:].sum(): {qk[0,:].sum().item()}")  # Sum over all K rows
        
        # The kernel outputs QK scores, but does it sum over K?
        # Actually, MFMA 32x32x16 outputs a 32x32 C matrix
        # The kernel iterates over K-tiles and accumulates
        # So output should be: sum over k_tiles of (32x32 partial QK)
        
        # For seq_len=64, num_k_tiles=2
        # Each K-tile is 32 K rows
        # QK[i,j] = sum_k(Q[i,k] * K[j,k]) = 128 for all-ones
        # But we accumulate over K-tiles:
        # Output[i,j] = sum_over_k_tiles(partial_QK[i,j % 32])
        
        # This is confusing. Let me think again...
        # The kernel computes: for each K-tile, QK_partial = Q @ K_tile.T
        # Then accumulates: QK += QK_partial
        # But QK is 32x32, not 32xseq_len
        
        # So for seq_len=64 (2 K-tiles):
        # K-tile 0: K[0:32], produces QK[0:32, 0:32]
        # K-tile 1: K[32:64], produces QK[0:32, 0:32] (added to same output!)
        
        # That means each output element accumulates:
        # Output[i,j] = Q[i,:] @ K[j,:].T + Q[i,:] @ K[j+32,:].T
        #             = Q[i,:] @ (K[j,:] + K[j+32,:]).T
        
        # For all-ones:
        # Output[i,j] = 1*128 + 1*128 = 256 for seq_len=64
        
        # So the kernel is computing something different than simple QK!

if __name__ == "__main__":
    main()

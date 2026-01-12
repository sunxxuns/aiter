#!/usr/bin/env python3
"""
Test BF16 Flash Attention Kernel as Baseline

Uses aiter API to test the BF16 kernel with same test patterns as FP8.
"""

import torch
import math


def test_bf16_kernel():
    """Test the BF16 kernel with similar tests as FP8"""
    
    try:
        from aiter.ops.mha import fmha_v3_fwd
    except ImportError:
        print("aiter not available, skipping BF16 baseline test")
        return True
    
    print("=" * 70)
    print("BF16 FLASH ATTENTION BASELINE TEST (via aiter API)")
    print("=" * 70)
    
    results = []
    scale = 1.0 / math.sqrt(128)
    
    for seq_len in [32, 64, 128, 256]:
        n_tiles = seq_len // 32
        print(f"\n📋 Testing seq_len={seq_len} ({n_tiles} tiles)")
        print("-" * 50)
        
        torch.manual_seed(42)
        
        # BF16 kernel expects [B, S, H, D] layout
        B, H, D = 1, 1, 128
        Q = torch.randn(B, 32, H, D, device='cuda', dtype=torch.bfloat16) * 0.5
        K = torch.randn(B, seq_len, H, D, device='cuda', dtype=torch.bfloat16) * 0.5
        
        # Test 1: V=1 identity test
        V = torch.ones(B, seq_len, H, D, device='cuda', dtype=torch.bfloat16)
        
        try:
            R, lse, _, _ = fmha_v3_fwd(
                Q, K, V,
                dropout_p=0.0,
                softmax_scale=scale,
                is_causal=False,
                window_size_left=-1,
                window_size_right=-1,
                return_softmax_lse=True,
                return_dropout_randval=False,
                how_v3_bf16_cvt=1,
            )
            
            R_mean = R.float().mean().item()
            v1_pass = abs(R_mean - 1.0) < 0.1
            print(f"  {'✅' if v1_pass else '❌'} V=1 identity: mean={R_mean:.4f} (expected ~1.0)")
            results.append(('V=1 identity', seq_len, v1_pass))
        except Exception as e:
            print(f"  ❌ V=1 identity: Error - {e}")
            results.append(('V=1 identity', seq_len, False))
            continue
        
        # Test 2: Tile 0 isolation
        V_iso = torch.zeros(B, seq_len, H, D, device='cuda', dtype=torch.bfloat16)
        V_iso[:, 0:32, :, :] = 1.0
        
        R_iso, _, _, _ = fmha_v3_fwd(
            Q, K, V_iso, 0.0, scale, False, -1, -1, True, False, 1
        )
        
        R_iso_mean = R_iso.float().mean().item()
        tile0_pass = R_iso_mean > 0.01
        print(f"  {'✅' if tile0_pass else '❌'} Tile 0 isolation: mean={R_iso_mean:.4f}")
        results.append(('Tile 0 isolation', seq_len, tile0_pass))
        
        # Test 3: Tile 1 isolation (if multi-tile)
        if n_tiles > 1:
            V_iso2 = torch.zeros(B, seq_len, H, D, device='cuda', dtype=torch.bfloat16)
            V_iso2[:, 32:64, :, :] = 1.0
            
            R_iso2, _, _, _ = fmha_v3_fwd(
                Q, K, V_iso2, 0.0, scale, False, -1, -1, True, False, 1
            )
            
            R_iso2_mean = R_iso2.float().mean().item()
            tile1_pass = R_iso2_mean > 0.01
            print(f"  {'✅' if tile1_pass else '❌'} Tile 1 isolation: mean={R_iso2_mean:.4f}")
            results.append(('Tile 1 isolation', seq_len, tile1_pass))
        
        # Test 4: Reference match
        V_rand = torch.randn(B, seq_len, H, D, device='cuda', dtype=torch.bfloat16) * 0.5
        
        R_rand, _, _, _ = fmha_v3_fwd(
            Q, K, V_rand, 0.0, scale, False, -1, -1, True, False, 1
        )
        
        # Reference calculation (simple attention)
        Q_2d = Q.squeeze(0).squeeze(1)  # [32, 128]
        K_2d = K.squeeze(0).squeeze(1)  # [seq, 128]
        V_2d = V_rand.squeeze(0).squeeze(1)  # [seq, 128]
        
        S = Q_2d.float() @ K_2d.float().T * scale
        P = torch.softmax(S, dim=1)
        R_ref = (P @ V_2d.float()).to(torch.bfloat16)
        
        R_rand_2d = R_rand.squeeze(0).squeeze(1)
        max_err = (R_rand_2d.float() - R_ref.float()).abs().max().item()
        ref_pass = max_err < 0.1
        print(f"  {'✅' if ref_pass else '❌'} Reference match: max_err={max_err:.4f}")
        results.append(('Reference match', seq_len, ref_pass))
        
        # Test 5: Element ratio consistency
        mask = R_ref.float().abs() > 0.01
        if mask.sum() > 10:
            ratios = R_rand_2d.float()[mask] / R_ref.float()[mask]
            ratio_std = ratios.std().item()
            ratio_mean = ratios.mean().item()
            ratio_pass = ratio_std < 0.3 and 0.7 < ratio_mean < 1.3
            print(f"  {'✅' if ratio_pass else '❌'} Ratio consistency: mean={ratio_mean:.3f}, std={ratio_std:.3f}")
            results.append(('Ratio consistency', seq_len, ratio_pass))
    
    # Summary
    passed = sum(1 for _, _, p in results if p)
    total = len(results)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ BF16 kernel passes all structured tests")
        print("   This validates our testing methodology!")
    else:
        failed = [(n, s) for n, s, p in results if not p]
        print(f"❌ Failed tests: {failed}")
    print("=" * 70)
    
    return all(p for _, _, p in results)


if __name__ == "__main__":
    test_bf16_kernel()

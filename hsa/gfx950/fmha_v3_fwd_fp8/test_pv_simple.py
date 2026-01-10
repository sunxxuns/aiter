#!/usr/bin/env python3
"""
Simple P×V simulation to understand the correct behavior.
"""

import torch
import numpy as np

def test_pv_math():
    """Test P×V math with simple examples."""
    print("="*60)
    print("P×V Math Verification")
    print("="*60)
    
    # Small example: P is 4x4, V is 4x4
    # O = P @ V
    
    # Case 1: P = softmax(uniform) = 0.25 each, V = 1
    # O[i,j] = sum_k(P[i,k] * V[k,j]) = sum_k(0.25 * 1) = 0.25 * 4 = 1.0
    P1 = torch.ones(4, 4) / 4  # Uniform softmax
    V1 = torch.ones(4, 4)
    O1 = P1 @ V1
    print(f"\nCase 1: P=uniform(0.25), V=1")
    print(f"Expected O = 1.0 everywhere")
    print(f"Actual O = {O1[0,0].item()}")
    
    # Case 2: P = uniform, V[k, d] = d
    # O[i,j] = sum_k(0.25 * j) = j * (0.25 * 4) = j
    V2 = torch.arange(4).float().unsqueeze(0).expand(4, 4)  # V[k,d] = d
    print(f"\nV2 (V[k,d]=d):\n{V2}")
    O2 = P1 @ V2
    print(f"\nCase 2: P=uniform, V[k,d]=d")
    print(f"Expected O[i,d] = d")
    print(f"Actual O[0,:] = {O2[0,:].tolist()}")
    
    # Case 3: P = uniform, V[k, d] = k
    # O[i,j] = sum_k(0.25 * k) = 0.25 * (0+1+2+3) = 0.25 * 6 = 1.5
    V3 = torch.arange(4).float().unsqueeze(1).expand(4, 4)  # V[k,d] = k
    print(f"\nV3 (V[k,d]=k):\n{V3}")
    O3 = P1 @ V3
    print(f"\nCase 3: P=uniform, V[k,d]=k")
    print(f"Expected O[i,d] = sum(k)/4 = 1.5")
    print(f"Actual O[0,:] = {O3[0,:].tolist()}")
    
    print("\n" + "="*60)
    print("Full MFMA Simulation (32x32x16)")
    print("="*60)
    
    # Simulate 32x32x16 MFMA for P×V
    # P: 32 Q rows × 16 K cols (reduced)
    # V: 16 K rows × 32 D cols
    # O: 32 Q rows × 32 D cols
    
    M, N, K = 32, 32, 16
    
    # P = uniform (1/K = 1/16 = 0.0625 per K position)
    P = torch.ones(M, K) / K
    # V = 1
    V = torch.ones(K, N)
    
    O_ref = P @ V
    print(f"\nP@V with P=uniform, V=1, K={K}:")
    print(f"Expected O = 1.0 (sum of 16 * (1/16) * 1)")
    print(f"Actual O[0,0] = {O_ref[0,0].item()}")
    
    # V[k,d] = d/32
    V_by_d = torch.arange(N).float().unsqueeze(0).expand(K, N) / N
    O_by_d = P @ V_by_d
    print(f"\nP@V with P=uniform, V[k,d]=d/N:")
    print(f"Expected O[q,d] = d/N")
    print(f"Actual O[0,:8] = {O_by_d[0,:8].tolist()}")
    print(f"Expected: {[d/N for d in range(8)]}")
    
    print("\n" + "="*60)
    print("FP8 Quantization Effects")
    print("="*60)
    
    # Check FP8 e4m3 representation
    v_float = torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=torch.float32)
    v_fp8 = v_float.to(torch.float8_e4m3fnuz)
    v_back = v_fp8.float()
    print(f"\nFP8 roundtrip:")
    for orig, back in zip(v_float.tolist(), v_back.tolist()):
        print(f"  {orig} -> {back} (error: {abs(orig-back)})")

if __name__ == "__main__":
    test_pv_math()

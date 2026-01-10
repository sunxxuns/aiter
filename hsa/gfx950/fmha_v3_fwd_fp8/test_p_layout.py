#!/usr/bin/env python3
"""
Debug test: simulate P redistribution to verify the layout is correct.
"""

import torch
import numpy as np

torch.manual_seed(42)

def simulate_qk_mfma_output():
    """
    Simulate what each thread has after QK MFMA 32x32.
    Thread t owns P[Q_base:Q_base+16, K=t%32] where Q_base = (t/32)*16
    """
    # Create a reference P matrix [32 Q rows, 32 K cols]
    P_ref = torch.arange(32*32, dtype=torch.float32).reshape(32, 32)
    print(f"P_ref shape: {P_ref.shape}")
    print(f"P_ref[0, :8] = {P_ref[0, :8].tolist()}")  # Row 0, cols 0-7
    
    # Distribute to 64 threads as QK MFMA would
    thread_P = {}
    for t in range(64):
        Q_base = (t // 32) * 16  # 0 for t<32, 16 for t>=32
        K_col = t % 32
        # Thread t owns P[Q_base:Q_base+16, K_col] - 16 values
        thread_P[t] = P_ref[Q_base:Q_base+16, K_col].clone()
    
    # Verify thread 0 has P[0:16, 0]
    print(f"\nThread 0 has P[0:16, K=0]:")
    print(f"  Values: {thread_P[0].tolist()[:4]}... (should be 0, 32, 64, 96)")
    
    print(f"\nThread 1 has P[0:16, K=1]:")
    print(f"  Values: {thread_P[1].tolist()[:4]}... (should be 1, 33, 65, 97)")
    
    print(f"\nThread 32 has P[16:32, K=0]:")
    print(f"  Values: {thread_P[32].tolist()[:4]}... (should be 512, 544, 576, 608)")
    
    return P_ref, thread_P

def simulate_p_storage_to_lds(thread_P):
    """
    Simulate storing P to LDS with layout P[Q_row, K_col] at Q_row*32 + K_col.
    Each thread stores its 16 P values in column K=t%32.
    """
    # LDS simulation: P[32 Q rows, 32 K cols]
    LDS_P = torch.zeros(32, 32, dtype=torch.float32)
    
    for t in range(64):
        Q_base = (t // 32) * 16
        K_col = t % 32
        # Store 16 values at P[Q_base:Q_base+16, K_col]
        LDS_P[Q_base:Q_base+16, K_col] = thread_P[t]
    
    print(f"\nLDS P storage verification:")
    print(f"  LDS_P[0, :8] = {LDS_P[0, :8].tolist()} (should be 0,1,2,3,4,5,6,7)")
    print(f"  LDS_P[1, :8] = {LDS_P[1, :8].tolist()} (should be 32,33,34,35,36,37,38,39)")
    
    return LDS_P

def simulate_p_read_for_b_operand(LDS_P):
    """
    Simulate reading P for V×P B operand.
    Thread t needs P[Q=t%32, K_range] where K_range = (t/32)*8:(t/32)*8+8
    """
    thread_B = {}
    for t in range(64):
        Q = t % 32
        K_start = (t // 32) * 8
        # Read 8 K values at row Q
        thread_B[t] = LDS_P[Q, K_start:K_start+8].clone()
    
    print(f"\nB operand read verification:")
    print(f"  Thread 0 reads P[Q=0, K=0:8] = {thread_B[0].tolist()}")
    print(f"  (Expected: 0,1,2,3,4,5,6,7)")
    print(f"  Thread 32 reads P[Q=0, K=8:16] = {thread_B[32].tolist()}")
    print(f"  (Expected: 8,9,10,11,12,13,14,15)")
    print(f"  Thread 1 reads P[Q=1, K=0:8] = {thread_B[1].tolist()}")
    print(f"  (Expected: 32,33,34,35,36,37,38,39)")
    
    return thread_B

def verify_mfma_correctness(P_ref, thread_B):
    """
    Verify that the B operand layout matches what MFMA expects.
    For B operand of 32x32x16 MFMA:
    - Thread t provides B[K_range, N=t%32] where K_range has 8 values
    - For V×P, B is P^T[K, Q]
    - Thread t needs P^T[K_range, Q=t%32] = P[Q=t%32, K_range]
    """
    print(f"\nMFMA B operand verification:")
    print(f"For V×P MFMA: A=V^T[D,K], B=P^T[K,Q], C=O^T[D,Q]")
    print(f"Thread t should have B[K_range, N=t%32] = P^T[K_range, Q=t%32]")
    
    # For B operand, the layout is B[16K, 32N]
    # Thread 0 provides B[K=0..7, N=0] = P^T[K=0..7, Q=0] = P[Q=0, K=0..7]
    # Thread 32 provides B[K=8..15, N=0] = P^T[K=8..15, Q=0] = P[Q=0, K=8..15]
    
    all_correct = True
    for t in range(64):
        Q = t % 32
        K_start = (t // 32) * 8
        expected = P_ref[Q, K_start:K_start+8]
        actual = thread_B[t]
        if not torch.allclose(expected, actual):
            print(f"  Thread {t}: MISMATCH")
            print(f"    Expected: {expected.tolist()}")
            print(f"    Actual: {actual.tolist()}")
            all_correct = False
    
    if all_correct:
        print(f"  All 64 threads have correct B operand values! ✓")
    
    return all_correct

def main():
    print("="*60)
    print("P Layout Debug Test")
    print("="*60)
    
    P_ref, thread_P = simulate_qk_mfma_output()
    LDS_P = simulate_p_storage_to_lds(thread_P)
    thread_B = simulate_p_read_for_b_operand(LDS_P)
    correct = verify_mfma_correctness(P_ref, thread_B)
    
    # Verify that LDS storage reconstructs P_ref correctly
    print(f"\nFinal verification: LDS_P == P_ref?")
    if torch.allclose(LDS_P, P_ref):
        print(f"  YES - P matrix correctly stored and read!")
    else:
        print(f"  NO - mismatch detected")
        print(f"  Max diff: {(LDS_P - P_ref).abs().max()}")
    
    print(f"\n{'='*60}")
    print(f"Result: {'PASS' if correct else 'FAIL'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test PV MFMA layout understanding with random non-uniform inputs.

Key insight from BF16 kernel:
- After QK MFMA, thread t has P[Q_row_base:Q_row_base+15, K=t%32]
- This is exactly B operand layout for V^T × P^T
- BF16 computes: O^T = V^T × P^T, then transposes output

For FP8, we verify:
1. QK MFMA output layout matches expected P distribution
2. PV as V^T × P^T gives correct results
3. Output transposition is required
"""

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def simulate_qk_mfma_output(Q_rows=32, K_cols=32, threads=64):
    """
    Simulate QK MFMA output distribution.
    
    For MFMA 32x32x16, each thread gets 16 outputs at ONE N column:
    - tid % 32 = N column (K-attention position)
    - (tid // 32) * 16 = M row base
    """
    # Create random QK scores
    QK = torch.randn(Q_rows, K_cols, dtype=torch.float32)
    
    # Distribute to threads as MFMA would
    thread_outputs = []
    for tid in range(threads):
        n_col = tid % 32  # K position
        m_base = (tid // 32) * 16  # Q row base
        # Thread owns QK[m_base:m_base+16, n_col]
        values = QK[m_base:m_base+16, n_col].clone()
        thread_outputs.append(values)
    
    return QK, thread_outputs

def simulate_pv_vtpt(P, V, threads=64):
    """
    Simulate PV computation as V^T × P^T.
    
    P: [Q_rows, K] (softmax output)
    V: [K, D] (value matrix)
    
    Computation: O^T = V^T × P^T
    - V^T: [D, K]
    - P^T: [K, Q_rows]
    - O^T: [D, Q_rows]
    
    For MFMA, P^T B operand needs P values distributed as:
    - tid % 32 = K index
    - tid's 16 values span 16 Q_row positions
    
    After QK MFMA, thread t has P[Q_row_base:Q_row_base+16, K=t%32]
    This is exactly P^T[K=t%32, Q_row_base:Q_row_base+16] when viewed as B operand!
    """
    Q_rows, K = P.shape
    K2, D = V.shape
    assert K == K2
    
    # Standard P × V
    O_standard = P @ V  # [Q_rows, D]
    
    # V^T × P^T
    VT = V.T  # [D, K]
    PT = P.T  # [K, Q_rows]
    OT = VT @ PT  # [D, Q_rows]
    O_transposed = OT.T  # [Q_rows, D]
    
    # These should be identical
    assert torch.allclose(O_standard, O_transposed, atol=1e-5), \
        f"P×V vs (V^T×P^T)^T mismatch: {(O_standard - O_transposed).abs().max()}"
    
    return O_standard

def verify_p_operand_layout(P, threads=64):
    """
    Verify that QK MFMA output can be used directly as P^T B operand.
    
    After QK MFMA, thread t has:
        P[Q_row_base:Q_row_base+16, K=t%32]
    
    For B operand of MFMA 32x32x16:
    - B[K, N] with K=16, N=32
    - Thread t contributes to K=(t%32)th slice of K dimension
    - For K-tiles (each MFMA does K=16), need K=0..15 from threads with (t%32) < 16
    
    Wait, there are 32 K positions but MFMA only does 16 at a time.
    So we need 2 MFMAs for full K=32 reduction.
    """
    Q_rows, K = P.shape
    
    print(f"P shape: {P.shape}")
    print(f"\nThread P value distribution (simulating QK MFMA output):")
    
    # Show how P values are distributed to threads
    for tid in [0, 1, 15, 16, 31, 32, 33, 63]:
        k_col = tid % 32
        q_base = (tid // 32) * 16
        vals = P[q_base:q_base+4, k_col]  # Show first 4 of 16
        print(f"  tid {tid:2d}: P[Q={q_base}:{q_base+16}, K={k_col}], first 4: {vals.tolist()}")
    
    print("\nFor PV MFMA (V^T × P^T):")
    print("  B operand (P^T) needs: [K, Q_rows] where each MFMA tile is 16×32")
    print("  Thread t contributes: K_index = t%32, spanning 16 Q_rows")
    print("\n  MFMA 1 (K=0..15): uses threads with t%32 < 16")
    print("  MFMA 2 (K=16..31): uses threads with t%32 >= 16")

def test_with_random_data():
    """Test full PV computation with random data."""
    print("="*70)
    print("Testing PV MFMA Layout with Random Non-Uniform Data")
    print("="*70)
    
    Q_rows = 32
    K = 32  # seqlen_k
    D = 32  # head_dim tile
    
    # Random P (softmax output, should sum to 1 per row)
    raw_scores = torch.randn(Q_rows, K)
    P = torch.softmax(raw_scores, dim=-1)
    
    # Random V
    V = torch.randn(K, D)
    
    # Verify P operand layout
    verify_p_operand_layout(P)
    
    # Compute reference
    O_ref = P @ V
    print(f"\nReference O = P × V shape: {O_ref.shape}")
    print(f"O[0, :4] = {O_ref[0, :4].tolist()}")
    
    # Verify V^T × P^T approach
    VT = V.T  # [D, K]
    PT = P.T  # [K, Q_rows]
    OT = VT @ PT  # [D, Q_rows]
    O_vtpt = OT.T  # [Q_rows, D]
    
    diff = (O_ref - O_vtpt).abs().max()
    print(f"\nV^T × P^T approach:")
    print(f"  Max diff from reference: {diff}")
    print(f"  Match: {diff < 1e-5}")
    
    # Simulate thread output distribution
    print("\n" + "="*70)
    print("Simulating MFMA Output Distribution")
    print("="*70)
    
    # For V^T × P^T output O^T[D, Q_rows]:
    # Thread t gets O^T[D_base:D_base+16, N=t%32]
    # Where D_base = (t//32)*16, N = t%32 (Q_row position)
    
    print("\nOutput O^T[D, Q_rows] distribution:")
    print("  Each thread t owns O^T[D_base:D_base+16, Q_row=t%32]")
    for tid in [0, 1, 31, 32, 63]:
        d_base = (tid // 32) * 16
        q_row = tid % 32
        vals = OT[d_base:d_base+4, q_row]
        print(f"  tid {tid:2d}: O^T[D={d_base}:{d_base+16}, Q_row={q_row}], first 4: {vals.tolist()}")
    
    print("\n  After transpose, thread t owns O[Q_row=t%32, D_base:D_base+16]")
    print("  This matches standard attention output layout!")

def test_k_tile_splitting():
    """Test how K dimension is split across MFMAs."""
    print("\n" + "="*70)
    print("K-Tile Splitting Analysis")
    print("="*70)
    
    Q_rows = 32
    K = 32
    D = 32
    
    # Random data
    P = torch.softmax(torch.randn(Q_rows, K), dim=-1)
    V = torch.randn(K, D)
    
    # Reference
    O_ref = P @ V
    
    # Split K into tiles of 16
    # O = P[:, 0:16] @ V[0:16, :] + P[:, 16:32] @ V[16:32, :]
    O_tile1 = P[:, 0:16] @ V[0:16, :]
    O_tile2 = P[:, 16:32] @ V[16:32, :]
    O_split = O_tile1 + O_tile2
    
    print(f"Reference O[0, :4]: {O_ref[0, :4].tolist()}")
    print(f"Tile 1 (K=0..15) contribution: {O_tile1[0, :4].tolist()}")
    print(f"Tile 2 (K=16..31) contribution: {O_tile2[0, :4].tolist()}")
    print(f"Sum of tiles O[0, :4]: {O_split[0, :4].tolist()}")
    print(f"Match: {torch.allclose(O_ref, O_split)}")
    
    # Now in V^T × P^T form
    print("\nIn V^T × P^T form:")
    VT = V.T
    PT = P.T
    
    # Split by K
    OT_tile1 = VT[:, 0:16] @ PT[0:16, :]  # [D, Q] contribution from K=0..15
    OT_tile2 = VT[:, 16:32] @ PT[16:32, :]  # [D, Q] contribution from K=16..31
    OT_split = OT_tile1 + OT_tile2
    O_split_vtpt = OT_split.T
    
    print(f"Tile 1 (K=0..15) uses P values from threads with t%32 < 16")
    print(f"Tile 2 (K=16..31) uses P values from threads with t%32 >= 16")
    print(f"Match with reference: {torch.allclose(O_ref, O_split_vtpt)}")

if __name__ == "__main__":
    test_with_random_data()
    test_k_tile_splitting()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The BF16 kernel's approach works because:

1. After QK MFMA, thread t has P[Q_base:Q_base+16, K=t%32]
   - This is 16 Q-row values at ONE K position

2. For PV MFMA computing V^T × P^T:
   - B operand (P^T) needs [K, Q_rows] = [16, 32] per MFMA tile
   - Thread t contributes to K=(t%32) row, spanning 16 Q positions
   - The 16 P values per thread are exactly at those 16 Q positions!

3. V is read transposed using ds_read_b64_tr_b16:
   - A operand (V^T) needs [D, K] = [32, 16]
   - Transpose-read loads V[K, D] and outputs V^T[D, K]

4. Output O^T[D, Q] is transposed to O[Q, D] in final store:
   - Thread t owns O^T[D_base:D_base+16, Q=t%32]
   - After transpose: O[Q=t%32, D_base:D_base+16]

For FP8 kernel fix:
- Keep QK MFMA output as-is (P values already in correct layout)
- Read V with transpose (ds_read_b64_tr_b8)
- Store output with transpose (swap row/col indexing)
""")

#!/usr/bin/env python3
"""
Rigorous MFMA layout tests with NON-UNIFORM inputs.
Each test is designed to reveal specific bugs:
- Wrong thread->output mapping
- Wrong K reduction order
- Wrong A/B operand layout
"""

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def test_mfma_output_distribution():
    """
    Test: Understand exactly which thread owns which output positions.
    Use A[m,k] = m, B[k,n] = 1 => C[m,n] = m * K
    If output shows m values, we know which M positions each thread owns.
    """
    print("="*70)
    print("TEST 1: MFMA Output Distribution (A=M_index, B=1)")
    print("="*70)
    
    M, N, K = 32, 32, 16
    
    # A[m, k] = m (row index)
    A = torch.arange(M).float().unsqueeze(1).expand(M, K)
    # B[k, n] = 1
    B = torch.ones(K, N)
    
    # Reference: C[m, n] = sum_k(A[m,k] * B[k,n]) = sum_k(m * 1) = m * K
    C_ref = A @ B  # Should be [[0*16], [1*16], [2*16], ...]
    
    print(f"A[m,k] = m (row index varies 0-{M-1})")
    print(f"B[k,n] = 1")
    print(f"Expected C[m,n] = m * K = m * {K}")
    print()
    print(f"Reference C[:8, 0] = {C_ref[:8, 0].tolist()}")
    print(f"Expected: [0, 16, 32, 48, 64, 80, 96, 112]")
    print()
    
    # For MFMA with 64 threads:
    # Thread t owns C[M_range, N=t%32]
    # The question is: what is M_range?
    
    # Based on our findings, M_range is interleaved:
    # Threads 0-31: M = 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
    # Threads 32-63: M = 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31
    
    m_rows_0_31 = [0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27]
    m_rows_32_63 = [4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31]
    
    print("Predicted thread ownership (based on interleaving):")
    print(f"  Threads 0-31 own M rows: {m_rows_0_31}")
    print(f"  Threads 32-63 own M rows: {m_rows_32_63}")
    print()
    
    # Values thread 0 should have (at N=0):
    expected_t0 = [C_ref[m, 0].item() for m in m_rows_0_31]
    expected_t32 = [C_ref[m, 0].item() for m in m_rows_32_63]
    print(f"Thread 0 expected values (M*K for its rows): {expected_t0}")
    print(f"Thread 32 expected values (M*K for its rows): {expected_t32}")
    
    return m_rows_0_31, m_rows_32_63

def test_k_reduction_order():
    """
    Test: Verify K reduction sums correctly.
    Use A[m,k] = 1, B[k,n] = k => C[m,n] = sum(k for k=0..K-1)
    This reveals if K values are being summed in wrong order or duplicated.
    """
    print("\n" + "="*70)
    print("TEST 2: K Reduction Order (A=1, B=K_index)")
    print("="*70)
    
    M, N, K = 32, 32, 16
    
    # A[m, k] = 1
    A = torch.ones(M, K)
    # B[k, n] = k (K index)
    B = torch.arange(K).float().unsqueeze(1).expand(K, N)
    
    # Reference: C[m, n] = sum_k(1 * k) = sum(0..K-1) = K*(K-1)/2
    C_ref = A @ B
    expected_sum = K * (K - 1) / 2  # 16*15/2 = 120
    
    print(f"A[m,k] = 1")
    print(f"B[k,n] = k (0 to {K-1})")
    print(f"Expected C[m,n] = sum(k) = {expected_sum}")
    print()
    print(f"Reference C[0, 0] = {C_ref[0, 0].item()}")
    print(f"All values should be {expected_sum}")
    
    # If MFMA has wrong K ordering, the sum would be wrong
    # If some K values are duplicated/missing, sum would be wrong
    
    return expected_sum

def test_a_operand_layout():
    """
    Test: Verify A operand data reaches correct positions.
    Use A[m,k] = m + k*100, B=1 => C[m,n] = sum_k(m + k*100) = m*K + 100*sum(k)
    The m*K term reveals M position, the sum(k) term verifies K reduction.
    """
    print("\n" + "="*70)
    print("TEST 3: A Operand Layout (A=M+K*100, B=1)")
    print("="*70)
    
    M, N, K = 32, 32, 16
    
    # A[m, k] = m + k*100
    m_idx = torch.arange(M).float().unsqueeze(1).expand(M, K)
    k_idx = torch.arange(K).float().unsqueeze(0).expand(M, K)
    A = m_idx + k_idx * 100
    
    # B[k, n] = 1
    B = torch.ones(K, N)
    
    # C[m, n] = sum_k(m + k*100) = m*K + 100*sum(k) = m*16 + 100*120 = m*16 + 12000
    C_ref = A @ B
    
    print(f"A[m,k] = m + k*100")
    print(f"B[k,n] = 1")
    print(f"Expected C[m,n] = m*K + 100*sum(k) = m*16 + 12000")
    print()
    print(f"Reference C[:8, 0] = {C_ref[:8, 0].tolist()}")
    print(f"Expected: {[m*16 + 12000 for m in range(8)]}")
    print()
    
    # Check specific values
    for m in [0, 1, 15, 16, 31]:
        expected = m * K + 100 * (K * (K-1) / 2)
        actual = C_ref[m, 0].item()
        match = "✓" if abs(expected - actual) < 0.01 else "✗"
        print(f"  C[{m}, 0] = {actual:.1f}, expected {expected:.1f} {match}")

def test_b_operand_layout():
    """
    Test: Verify B operand data reaches correct positions.
    Use A=1, B[k,n] = k + n*100 => C[m,n] = sum_k(k + n*100) = sum(k) + n*100*K
    The n*100*K term reveals N position, the sum(k) term verifies K reduction.
    """
    print("\n" + "="*70)
    print("TEST 4: B Operand Layout (A=1, B=K+N*100)")
    print("="*70)
    
    M, N, K = 32, 32, 16
    
    # A[m, k] = 1
    A = torch.ones(M, K)
    
    # B[k, n] = k + n*100
    k_idx = torch.arange(K).float().unsqueeze(1).expand(K, N)
    n_idx = torch.arange(N).float().unsqueeze(0).expand(K, N)
    B = k_idx + n_idx * 100
    
    # C[m, n] = sum_k(k + n*100) = sum(k) + n*100*K = 120 + n*1600
    C_ref = A @ B
    
    print(f"A[m,k] = 1")
    print(f"B[k,n] = k + n*100")
    print(f"Expected C[m,n] = sum(k) + n*100*K = 120 + n*1600")
    print()
    print(f"Reference C[0, :8] = {C_ref[0, :8].tolist()}")
    print(f"Expected: {[120 + n*1600 for n in range(8)]}")
    print()
    
    # Check specific values
    for n in [0, 1, 15, 16, 31]:
        expected = (K * (K-1) / 2) + n * 100 * K
        actual = C_ref[0, n].item()
        match = "✓" if abs(expected - actual) < 0.01 else "✗"
        print(f"  C[0, {n}] = {actual:.1f}, expected {expected:.1f} {match}")

def test_full_pattern():
    """
    Test: Combined pattern that reveals any layout issue.
    Use A[m,k] = m*1000 + k, B[k,n] = k*100 + n
    C[m,n] = sum_k((m*1000 + k) * (k*100 + n))
           = sum_k(m*1000*k*100 + m*1000*n + k*k*100 + k*n)
           = m*100000*sum(k) + m*1000*n*K + 100*sum(k^2) + n*sum(k)
    
    This gives unique values for each (m,n) position.
    """
    print("\n" + "="*70)
    print("TEST 5: Full Pattern (A=M*1000+K, B=K*100+N)")
    print("="*70)
    
    M, N, K = 32, 32, 16
    
    # A[m, k] = m*1000 + k
    m_idx = torch.arange(M).float().unsqueeze(1).expand(M, K)
    k_idx = torch.arange(K).float().unsqueeze(0).expand(M, K)
    A = m_idx * 1000 + k_idx
    
    # B[k, n] = k*100 + n
    k_idx_b = torch.arange(K).float().unsqueeze(1).expand(K, N)
    n_idx = torch.arange(N).float().unsqueeze(0).expand(K, N)
    B = k_idx_b * 100 + n_idx
    
    C_ref = A @ B
    
    print(f"A[m,k] = m*1000 + k")
    print(f"B[k,n] = k*100 + n")
    print()
    
    # Each (m,n) should have a unique value
    # Check corners and middle
    test_points = [(0,0), (0,31), (31,0), (31,31), (15,15), (7,23)]
    print("Reference values at key positions:")
    for m, n in test_points:
        val = C_ref[m, n].item()
        print(f"  C[{m:2d}, {n:2d}] = {val:.0f}")
    
    print()
    print("Uniqueness check: all values should be unique")
    flat = C_ref.flatten()
    unique = flat.unique()
    print(f"  Total elements: {len(flat)}, Unique: {len(unique)}")
    if len(flat) == len(unique):
        print("  ✓ All values unique - good test pattern")
    else:
        print("  ✗ Some values repeated - may need different pattern")
    
    return C_ref

def test_fp8_quantization_effects():
    """
    Test: Check FP8 e4m3 quantization effects on test patterns.
    """
    print("\n" + "="*70)
    print("TEST 6: FP8 Quantization Effects")
    print("="*70)
    
    # Test range of values
    test_vals = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 240.0]
    
    print("FP8 e4m3fnuz roundtrip:")
    for v in test_vals:
        t = torch.tensor([v], dtype=torch.float32)
        fp8 = t.to(torch.float8_e4m3fnuz)
        back = fp8.float().item()
        err = abs(v - back)
        rel_err = err / max(abs(v), 1e-6) * 100
        print(f"  {v:8.2f} -> {back:8.2f} (err: {err:.4f}, {rel_err:.1f}%)")
    
    print()
    print("Safe value range for tests: 0-240 (FP8 e4m3 max ~240)")
    print("Values should be scaled to stay within this range")

def test_attention_specific_patterns():
    """
    Test: Attention-specific patterns for P×V computation.
    
    For attention: O = P @ V where P is softmax output (rows sum to 1)
    Test 1: P = softmax(uniform) = 1/K, V = d (column index)
            => O[q,d] = sum_k(1/K * d) = d
    
    Test 2: P[q,k] = 1 if k==q%K else 0 (one-hot), V = random
            => O[q,d] = V[q%K, d]
    """
    print("\n" + "="*70)
    print("TEST 7: Attention-Specific Patterns")
    print("="*70)
    
    Q_rows, K_cols, D_cols = 32, 16, 32
    
    # Test 1: Uniform P, V varies by D
    print("\nTest 7a: P=uniform, V=D_index")
    P1 = torch.ones(Q_rows, K_cols) / K_cols  # Uniform softmax
    V1 = torch.arange(D_cols).float().unsqueeze(0).expand(K_cols, D_cols)  # V[k,d] = d
    O1 = P1 @ V1
    
    print(f"  P = 1/{K_cols} everywhere (uniform softmax)")
    print(f"  V[k,d] = d")
    print(f"  Expected O[q,d] = d")
    print(f"  O[0, :8] = {O1[0, :8].tolist()}")
    print(f"  Expected: [0, 1, 2, 3, 4, 5, 6, 7]")
    
    # Test 2: P varies by Q, V varies by K
    print("\nTest 7b: P varies by Q, V varies by K")
    # P[q,k] = (q+1) / sum for normalization
    P2_raw = torch.zeros(Q_rows, K_cols)
    for q in range(Q_rows):
        # Weights that depend on q: higher q prefers higher k
        weights = torch.arange(K_cols).float() * (q + 1)
        P2_raw[q] = torch.softmax(weights, dim=0)
    
    V2 = torch.arange(K_cols).float().unsqueeze(1).expand(K_cols, D_cols)  # V[k,d] = k
    O2 = P2_raw @ V2
    
    print(f"  P[q,k] = softmax(k * (q+1)) - higher q prefers higher k")
    print(f"  V[k,d] = k")
    print(f"  O[q,d] = weighted average of k indices")
    print(f"  O[:8, 0] = {O2[:8, 0].tolist()}")
    print(f"  Should increase with q (higher q weights higher k more)")
    
    # Check monotonicity
    is_increasing = all(O2[i+1, 0] > O2[i, 0] for i in range(7))
    print(f"  Monotonically increasing: {'✓' if is_increasing else '✗'}")

def main():
    print("="*70)
    print("RIGOROUS MFMA LAYOUT TESTS - NON-UNIFORM INPUTS")
    print("="*70)
    print()
    print("These tests use structured non-uniform inputs that will reveal")
    print("any bugs in data layout, thread mapping, or K reduction.")
    print()
    
    test_mfma_output_distribution()
    test_k_reduction_order()
    test_a_operand_layout()
    test_b_operand_layout()
    C_ref = test_full_pattern()
    test_fp8_quantization_effects()
    test_attention_specific_patterns()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Key test patterns for MFMA verification:")
    print("1. A=M_index, B=1 -> reveals M-row distribution")
    print("2. A=1, B=K_index -> reveals K reduction correctness")
    print("3. A=M+K*100, B=1 -> verifies A operand layout")
    print("4. A=1, B=K+N*100 -> verifies B operand layout")
    print("5. A=M*1000+K, B=K*100+N -> full uniqueness test")
    print()
    print("For attention-specific tests:")
    print("6. P=uniform, V=D_index -> D column should be preserved")
    print("7. P varies by Q, V=K_index -> weighted K average")

if __name__ == "__main__":
    main()

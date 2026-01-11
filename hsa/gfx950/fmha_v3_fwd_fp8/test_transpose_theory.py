#!/usr/bin/env python3
"""
Test the transpose hypothesis:

If the kernel computes S^T = K @ Q^T instead of S = Q @ K^T:
- S^T[key, query] - each column is one query's attention to all keys
- MFMA thread t holds S^T[:, query=t%32] 
- Summing across VGPRs (different keys) for one thread = per-query sum = ROW-WISE!

The output would then be:
- P^T[key, query] = softmax(S^T, dim=0)  # softmax over keys (column-wise in S^T = row-wise in S)
- O^T = V^T @ P^T  # or equivalently O = P @ V

Let's verify this gives correct attention.
"""

import torch
import math

def attention_standard(Q, K, V):
    """Standard attention: S = Q @ K^T, row-wise softmax"""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = Q @ K.T
    P = torch.softmax(S * scale, dim=1)  # row-wise (over keys)
    O = P @ V
    return O, S, P

def attention_transposed(Q, K, V):
    """
    Transposed computation: S^T = K @ Q^T
    Then softmax over COLUMNS of S^T (which are rows of S)
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    St = K @ Q.T  # S^T[key, query]
    
    # Softmax over keys (dim=0 in S^T = dim=1 in S)
    Pt = torch.softmax(St * scale, dim=0)  # column-wise in S^T = row-wise in S
    
    # O^T = V^T @ P^T  =>  O = (V^T @ P^T)^T = P @ V
    # Or equivalently: Ot = V.T @ Pt, then O = Ot.T
    Ot = V.T @ Pt  # [HD, query]
    O = Ot.T       # [query, HD]
    
    return O, St, Pt

print("=" * 70)
print("TRANSPOSE HYPOTHESIS TEST")
print("=" * 70)

SEQ, HD = 32, 128
torch.manual_seed(42)

# Diagonal-dominant pattern
Q = torch.zeros(SEQ, HD, device='cuda')
K = torch.zeros(SEQ, HD, device='cuda')
V = torch.zeros(SEQ, HD, device='cuda')

for i in range(SEQ):
    Q[i, i % HD] = 2.0
    K[i, i % HD] = 2.0
    Q[i, :] += 0.1
    K[i, :] += 0.1
    V[i, :] = float(i) / SEQ

O_std, S_std, P_std = attention_standard(Q, K, V)
O_trans, St, Pt = attention_transposed(Q, K, V)

print(f"\nS_std shape: {S_std.shape} (query × key)")
print(f"S^T shape:   {St.shape} (key × query)")

print(f"\nS_std[0, :8]:  {S_std[0, :8].tolist()}")
print(f"S^T[:8, 0]:    {St[:8, 0].tolist()}")
print(f"  → Same values, transposed ✓" if torch.allclose(S_std[0, :8], St[:8, 0]) else "  → Different!")

print(f"\nP_std[0, :8] (row 0 attention):  {P_std[0, :8].tolist()}")
print(f"P^T[:8, 0] (col 0 = row 0 of P): {Pt[:8, 0].tolist()}")
print(f"  → Same values ✓" if torch.allclose(P_std[0, :8], Pt[:8, 0]) else "  → Different!")

print(f"\nP_std[0,:].sum() (should be 1.0): {P_std[0,:].sum().item():.6f}")
print(f"P^T[:,0].sum() (should be 1.0):   {Pt[:,0].sum().item():.6f}")

print(f"\nO_standard[0,:4]:   {O_std[0,:4].tolist()}")
print(f"O_transposed[0,:4]: {O_trans[0,:4].tolist()}")

diff = (O_std - O_trans).abs().max().item()
print(f"\nMax diff between standard and transposed: {diff:.10f}")

if diff < 1e-5:
    print("\n✅ TRANSPOSED COMPUTATION GIVES SAME RESULT!")
    print("""
KEY INSIGHT:
If the kernel computes S^T = K @ Q^T instead of S = Q @ K^T:
- Thread t holds ALL keys' scores for ONE query (column t of S^T)
- Summing VGPRs v32-v47 = summing across 16 key rows for that query
- This IS row-wise softmax (per-query normalization)!

The BF16 kernel achieves row-wise softmax by computing the transpose!
""")
else:
    print("\n❌ Results differ - hypothesis needs refinement")

# Verify with random input too
print("\n" + "=" * 70)
print("RANDOM INPUT VERIFICATION")
print("=" * 70)

torch.manual_seed(123)
Q = torch.randn(SEQ, HD, device='cuda')
K = torch.randn(SEQ, HD, device='cuda')
V = torch.randn(SEQ, HD, device='cuda')

O_std, _, _ = attention_standard(Q, K, V)
O_trans, _, _ = attention_transposed(Q, K, V)

diff = (O_std - O_trans).abs().max().item()
print(f"Max diff: {diff:.10f}")
print("✅ Verified!" if diff < 1e-5 else "❌ Differs")

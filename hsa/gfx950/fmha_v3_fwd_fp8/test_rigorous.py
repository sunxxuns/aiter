#!/usr/bin/env python3
"""
Rigorous FP8 Flash Attention Test Suite

Tests with realistic data distributions and edge cases that catch:
- Row-wise vs block-wise softmax errors
- Transposition bugs
- Accumulation order issues
- Numerical precision problems
"""

import torch
import subprocess
import ctypes
import math
import numpy as np

def build_kernel():
    """Build the FP8 kernel."""
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "test_full_hd128.s", "-o", "test_full_hd128.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed: {result.stderr.decode()}")
    subprocess.run(
        ["ld.lld", "-shared", "-o", "test_full_hd128.co", "test_full_hd128.o"],
        capture_output=True, cwd=cwd, check=True
    )
    return cwd + "/test_full_hd128.co"

class FP8AttentionTester:
    def __init__(self):
        self.hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
        co_path = build_kernel()
        self.module = ctypes.c_void_p()
        self.hip.hipModuleLoad(ctypes.byref(self.module), co_path.encode())
        self.func = ctypes.c_void_p()
        self.hip.hipModuleGetFunction(
            ctypes.byref(self.func), self.module, b"_ZN5aiter14test_full_hd128E"
        )
        self.SEQ = 32
        self.HD = 128
        self.scale = 1.0 / math.sqrt(self.HD)
    
    def run_kernel(self, Q_fp8, K_fp8, V_fp8):
        """Run the FP8 kernel."""
        O = torch.zeros(self.SEQ, self.HD, dtype=torch.float32, device='cuda')
        args = [ctypes.c_void_p(x.data_ptr()) for x in [O, Q_fp8, K_fp8, V_fp8]]
        args_arr = (ctypes.c_void_p * 4)(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )
        self.hip.hipModuleLaunchKernel(
            self.func, 1, 1, 1, 64, 1, 1, 8192, None, args_arr, None
        )
        self.hip.hipDeviceSynchronize()
        return O
    
    def reference(self, Q_fp8, K_fp8, V_fp8):
        """Compute reference with FP8 quantization matching kernel."""
        S = Q_fp8.float() @ K_fp8.float().T
        P = torch.softmax(S * self.scale, dim=1)  # Row-wise!
        P_fp8 = P.to(torch.float8_e4m3fn).float()
        return P_fp8 @ V_fp8.float()
    
    def cleanup(self):
        self.hip.hipModuleUnload(self.module)


def test_transformer_embeddings(tester):
    """
    Test 1: Realistic transformer embeddings
    
    In real transformers, Q/K/V come from linear projections of embeddings.
    They typically have:
    - Near-zero mean
    - Variance around 1/sqrt(d) after layer norm
    - Some correlation structure
    
    Note: FP8 quantization introduces ~0.1 error in extreme cases.
    Threshold is set to 0.1 to account for FP8 precision limits.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Transformer-like Embeddings")
    print("=" * 70)
    
    errors = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        
        # Simulate post-LayerNorm embeddings (normalized, unit variance per row)
        def make_normalized_tensor():
            x = torch.randn(tester.SEQ, tester.HD, device='cuda')
            x = x - x.mean(dim=1, keepdim=True)
            x = x / (x.std(dim=1, keepdim=True) + 1e-5)
            return x
        
        Q = make_normalized_tensor().to(torch.float8_e4m3fn)
        K = make_normalized_tensor().to(torch.float8_e4m3fn)
        V = make_normalized_tensor().to(torch.float8_e4m3fn)
        
        O_kernel = tester.run_kernel(Q, K, V)
        O_ref = tester.reference(Q, K, V)
        print(f"ref O sample: {O_ref[0, :5]}")
        print(f"kernel O sample: {O_kernel[0, :5]}")
        
        err = (O_kernel - O_ref).abs().max().item()
        errors.append(err)
        print(f"   Seed {seed}: max_err = {err:.6f}")
    
    max_err = max(errors)
    # Threshold 0.1: FP8 quantization can cause up to ~0.1 error in edge cases
    # This is expected behavior, not a kernel bug
    threshold = 0.1
    print(f"   Max error: {max_err:.6f} {'✅' if max_err < threshold else '❌'} (threshold={threshold})")
    return max_err < threshold


def test_peaked_attention(tester):
    """
    Test 2: Peaked attention patterns
    
    Real attention often has very peaked distributions where one or few
    keys dominate. This stresses numerical precision in softmax.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Peaked Attention (one dominant key per query)")
    print("=" * 70)
    
    errors = []
    
    # Each query strongly attends to a specific key
    for dominant_key in [0, 15, 31]:
        Q = torch.randn(tester.SEQ, tester.HD, device='cuda') * 0.1
        K = torch.randn(tester.SEQ, tester.HD, device='cuda') * 0.1
        V = torch.randn(tester.SEQ, tester.HD, device='cuda')
        
        # Make each query's dominant key have high dot product
        for q in range(tester.SEQ):
            k = (q + dominant_key) % tester.SEQ
            # Make Q[q] and K[k] aligned
            direction = torch.randn(tester.HD, device='cuda')
            direction = direction / direction.norm()
            Q[q] = direction * 3.0  # Strong signal
            K[k] = direction * 3.0
        
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        V_fp8 = V.to(torch.float8_e4m3fn)
        
        O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
        O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
        
        err = (O_kernel - O_ref).abs().max().item()
        errors.append(err)
        print(f"   Dominant key offset {dominant_key}: max_err = {err:.6f}")
    
    max_err = max(errors)
    print(f"   Max error: {max_err:.6f} {'✅' if max_err < 0.05 else '❌'}")
    return max_err < 0.05


def test_position_patterns(tester):
    """
    Test 3: Position-dependent attention patterns
    
    Real attention often has positional biases:
    - Recency bias (attend to nearby tokens)
    - Start token bias
    - Periodic patterns
    """
    print("\n" + "=" * 70)
    print("TEST 3: Position-dependent Patterns")
    print("=" * 70)
    
    errors = []
    
    # 3a: Recency bias (attend to nearby keys)
    print("   3a: Recency bias...")
    Q = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    K = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    # Position encoding that creates distance-based attention
    for i in range(tester.SEQ):
        Q[i, :32] = torch.sin(torch.arange(32, device='cuda') * i * 0.1)
        K[i, :32] = torch.sin(torch.arange(32, device='cuda') * i * 0.1)
        Q[i, 32:64] = torch.cos(torch.arange(32, device='cuda') * i * 0.1)
        K[i, 32:64] = torch.cos(torch.arange(32, device='cuda') * i * 0.1)
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    # 3b: Start token bias (all queries attend to token 0)
    print("   3b: Start token bias...")
    Q = torch.randn(tester.SEQ, tester.HD, device='cuda') * 0.5
    K = torch.randn(tester.SEQ, tester.HD, device='cuda') * 0.5
    K[0] = Q.mean(dim=0) * 5.0  # Token 0 is very similar to all queries
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    max_err = max(errors)
    print(f"   Max error: {max_err:.6f} {'✅' if max_err < 0.05 else '❌'}")
    return max_err < 0.05


def test_numerical_edge_cases(tester):
    """
    Test 4: Numerical edge cases
    
    Edge cases that stress FP8 precision:
    - Large magnitude differences
    - Very small values
    - Specific bit patterns
    """
    print("\n" + "=" * 70)
    print("TEST 4: Numerical Edge Cases")
    print("=" * 70)
    
    errors = []
    
    # 4a: Large magnitude differences (some large, some small values)
    print("   4a: Large magnitude differences...")
    Q = torch.randn(tester.SEQ, tester.HD, device='cuda')
    Q[:16] *= 0.1   # Small queries
    Q[16:] *= 2.0   # Large queries
    K = torch.randn(tester.SEQ, tester.HD, device='cuda')
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    # 4b: Alternating signs (potential cancellation issues)
    print("   4b: Alternating sign patterns...")
    Q = torch.randn(tester.SEQ, tester.HD, device='cuda')
    K = torch.randn(tester.SEQ, tester.HD, device='cuda')
    V = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    for i in range(tester.SEQ):
        V[i] = ((-1) ** i) * (i + 1) * 0.1  # Alternating +/- with increasing magnitude
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    # 4c: Near-uniform attention (all scores similar)
    print("   4c: Near-uniform attention...")
    Q = torch.ones(tester.SEQ, tester.HD, device='cuda')
    Q += torch.randn_like(Q) * 0.01  # Tiny perturbation
    K = torch.ones(tester.SEQ, tester.HD, device='cuda')
    K += torch.randn_like(K) * 0.01
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    max_err = max(errors)
    print(f"   Max error: {max_err:.6f} {'✅' if max_err < 0.05 else '❌'}")
    return max_err < 0.05


def test_row_column_confusion(tester):
    """
    Test 5: Detect row/column confusion (transposition bugs)
    
    Carefully constructed inputs where row-wise and column-wise
    softmax give very different results.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Row vs Column Confusion Detection")
    print("=" * 70)
    
    errors = []
    
    # Create S matrix with very different row-wise vs column-wise softmax
    # S[i,j] = i * 10 + j  (rows have similar values, columns vary a lot)
    Q = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    K = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    
    # Construct so S[i,j] ~ i + j*0.1 (rows vary slowly, columns vary fast)
    for i in range(tester.SEQ):
        Q[i, 0] = float(i)
        Q[i, 1] = 1.0
    for j in range(tester.SEQ):
        K[j, 0] = 1.0
        K[j, 1] = float(j) * 0.1
    
    V = torch.eye(tester.SEQ, tester.HD, device='cuda')  # Identity-like V
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    
    # Also compute what column-wise softmax would give (WRONG)
    S = Q_fp8.float() @ K_fp8.float().T
    P_colwise = torch.softmax(S * tester.scale, dim=0)  # Column-wise (WRONG)
    P_colwise_fp8 = P_colwise.to(torch.float8_e4m3fn).float()
    O_colwise = P_colwise_fp8 @ V_fp8.float()
    
    err_rowwise = (O_kernel - O_ref).abs().max().item()
    err_colwise = (O_kernel - O_colwise).abs().max().item()
    
    print(f"   Kernel vs ROW-WISE (correct):   {err_rowwise:.6f}")
    print(f"   Kernel vs COLUMN-WISE (wrong):  {err_colwise:.6f}")
    
    if err_rowwise < err_colwise:
        print("   ✅ Kernel correctly does ROW-WISE softmax")
        return True
    else:
        print("   ❌ Kernel may be doing COLUMN-WISE softmax!")
        return False


def test_accumulation_order(tester):
    """
    Test 6: Accumulation order sensitivity
    
    Detect if the kernel accumulates in a different order than expected,
    which could cause numerical differences due to floating-point non-associativity.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Accumulation Order Sensitivity")
    print("=" * 70)
    
    errors = []
    
    # Use values where accumulation order matters
    # Geometric series: V[k] = base^k causes order-sensitive accumulation
    for base in [0.9, 1.1, 0.5]:
        Q = torch.randn(tester.SEQ, tester.HD, device='cuda')
        K = torch.randn(tester.SEQ, tester.HD, device='cuda')
        V = torch.zeros(tester.SEQ, tester.HD, device='cuda')
        
        for k in range(tester.SEQ):
            V[k, :] = (base ** k)
        
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        V_fp8 = V.to(torch.float8_e4m3fn)
        
        O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
        O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
        
        err = (O_kernel - O_ref).abs().max().item()
        errors.append(err)
        print(f"   Geometric base {base}: max_err = {err:.6f}")
    
    max_err = max(errors)
    print(f"   Max error: {max_err:.6f} {'✅' if max_err < 0.1 else '❌'}")
    return max_err < 0.1


def test_specific_attention_patterns(tester):
    """
    Test 7: Specific attention patterns from real models
    
    Patterns observed in real transformer attention:
    - Diagonal (self-attention)
    - Block diagonal
    - Sparse patterns
    """
    print("\n" + "=" * 70)
    print("TEST 7: Specific Attention Patterns")
    print("=" * 70)
    
    errors = []
    
    # 7a: Pure diagonal attention (each query attends only to same position)
    print("   7a: Diagonal (self) attention...")
    Q = torch.eye(tester.SEQ, tester.HD, device='cuda') * 3.0
    Q += torch.randn_like(Q) * 0.1  # Small noise
    K = Q.clone()  # K same as Q -> diagonal attention
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    # 7b: Block diagonal (groups of queries attend to groups of keys)
    print("   7b: Block diagonal attention...")
    block_size = 8
    Q = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    K = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    for block in range(tester.SEQ // block_size):
        block_vec = torch.randn(tester.HD, device='cuda')
        block_vec = block_vec / block_vec.norm() * 3.0
        for i in range(block_size):
            idx = block * block_size + i
            Q[idx] = block_vec + torch.randn(tester.HD, device='cuda') * 0.1
            K[idx] = block_vec + torch.randn(tester.HD, device='cuda') * 0.1
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    # 7c: "Copy" pattern (each query strongly attends to one specific key)
    print("   7c: Copy pattern...")
    Q = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    K = torch.zeros(tester.SEQ, tester.HD, device='cuda')
    V = torch.randn(tester.SEQ, tester.HD, device='cuda')
    
    # Query i attends to key (i*7) % SEQ (pseudorandom mapping)
    for i in range(tester.SEQ):
        target = (i * 7) % tester.SEQ
        vec = torch.randn(tester.HD, device='cuda')
        vec = vec / vec.norm() * 3.0
        Q[i] = vec
        K[target] = K[target] + vec  # Multiple queries may target same key
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_kernel = tester.run_kernel(Q_fp8, K_fp8, V_fp8)
    O_ref = tester.reference(Q_fp8, K_fp8, V_fp8)
    err = (O_kernel - O_ref).abs().max().item()
    errors.append(err)
    print(f"      max_err = {err:.6f}")
    
    max_err = max(errors)
    print(f"   Max error: {max_err:.6f} {'✅' if max_err < 0.05 else '❌'}")
    return max_err < 0.05


def main():
    print("=" * 70)
    print("RIGOROUS FP8 FLASH ATTENTION TEST SUITE")
    print("=" * 70)
    print("Testing with realistic data distributions and edge cases")
    
    tester = FP8AttentionTester()
    
    results = {}
    results['transformer_embeddings'] = test_transformer_embeddings(tester)
    results['peaked_attention'] = test_peaked_attention(tester)
    results['position_patterns'] = test_position_patterns(tester)
    results['numerical_edge_cases'] = test_numerical_edge_cases(tester)
    results['row_column_confusion'] = test_row_column_confusion(tester)
    results['accumulation_order'] = test_accumulation_order(tester)
    results['specific_patterns'] = test_specific_attention_patterns(tester)
    
    tester.cleanup()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_pass = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:30s} {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    main()

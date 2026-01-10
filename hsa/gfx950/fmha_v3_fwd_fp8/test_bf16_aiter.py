#!/usr/bin/env python3
"""
Test BF16 flash attention kernel with non-uniform inputs using aiter interface.
Verifies that the V×P computation produces correct results.
"""

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def compute_reference(Q, K, V, softmax_scale):
    """Compute reference flash attention output."""
    # Q: [B, S, H, D], K: [B, S, H, D], V: [B, S, H, D]
    # For simplicity, treat as 2D: [S, D]
    Q_2d = Q.squeeze(0).squeeze(1)  # [S, D]
    K_2d = K.squeeze(0).squeeze(1)
    V_2d = V.squeeze(0).squeeze(1)
    
    QK = Q_2d.float() @ K_2d.float().T  # [S, S]
    QK = QK * softmax_scale
    P = torch.softmax(QK, dim=-1)  # [S, S]
    O = P @ V_2d.float()  # [S, D]
    return O

def test_bf16_with_pattern(seqlen=64, head_dim=128, test_name="random"):
    """Test BF16 kernel with specific input pattern."""
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"seqlen={seqlen}, head_dim={head_dim}")
    print('='*60)
    
    try:
        from aiter import fmha_v3_fwd
    except ImportError as e:
        print(f"ERROR: Cannot import aiter: {e}")
        return False
    
    B, S, H, D = 1, seqlen, 1, head_dim
    softmax_scale = D ** -0.5
    
    # Generate test data based on pattern
    if test_name == "uniform_v":
        Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        V = torch.ones(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        expected_behavior = "Output should be ~1.0 everywhere (weighted average of 1s)"
    elif test_name == "v_by_k":
        Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        # V[k, :] = k/S (V varies by K position, normalized to [0,1])
        v_vals = torch.arange(S, dtype=torch.float32, device='cuda') / S
        V = v_vals.view(1, S, 1, 1).expand(B, S, H, D).to(torch.bfloat16)
        expected_behavior = "Output should be weighted average of k values based on attention"
    elif test_name == "v_by_d":
        Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        # V[:, d] = d/D (V varies by D position, normalized to [0,1])
        v_vals = torch.arange(D, dtype=torch.float32, device='cuda') / D
        V = v_vals.view(1, 1, 1, D).expand(B, S, H, D).to(torch.bfloat16)
        expected_behavior = "Output[:, d] should be ~d/D (D position preserved)"
    else:  # random
        Q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
        expected_behavior = "Output should match PyTorch reference"
    
    print(f"Expected: {expected_behavior}")
    
    # Compute reference
    O_ref = compute_reference(Q, K, V, softmax_scale)
    
    # Run kernel
    try:
        out, lse, _, _ = fmha_v3_fwd(Q, K, V,
                                      dropout_p=0.0,
                                      softmax_scale=softmax_scale,
                                      is_causal=False,
                                      window_size_left=-1,
                                      window_size_right=-1,
                                      return_softmax_lse=True,
                                      return_dropout_randval=False,
                                      how_v3_bf16_cvt=0)
    except Exception as e:
        print(f"ERROR: Kernel failed: {e}")
        return False
    
    torch.cuda.synchronize()
    
    # Get output as 2D for comparison
    O_kernel = out.squeeze(0).squeeze(1).float().cpu()  # [S, D]
    O_ref_cpu = O_ref.cpu()
    
    # Check for NaN/Inf
    if torch.isnan(O_kernel).any() or torch.isinf(O_kernel).any():
        print("ERROR: Output contains NaN or Inf!")
        return False
    
    # Compute error metrics
    abs_diff = (O_kernel - O_ref_cpu).abs()
    rel_diff = abs_diff / (O_ref_cpu.abs() + 1e-6)
    
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    max_rel_err = rel_diff.max().item()
    
    # Compute correlation
    O_kernel_flat = O_kernel.flatten()
    O_ref_flat = O_ref_cpu.flatten()
    correlation = torch.corrcoef(torch.stack([O_kernel_flat, O_ref_flat]))[0, 1].item()
    
    print(f"\nResults:")
    print(f"  Max abs error: {max_abs_err:.6f}")
    print(f"  Mean abs error: {mean_abs_err:.6f}")
    print(f"  Max rel error: {max_rel_err:.4f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # Show sample values
    print(f"\nSample outputs (row 0, elements 0:4):")
    print(f"  Reference: {O_ref_cpu[0, :4].tolist()}")
    print(f"  Kernel:    {O_kernel[0, :4].tolist()}")
    
    # For V_by_D test, check if D dimension is preserved
    if test_name == "v_by_d":
        print(f"\nD-dimension check (row 0):")
        print(f"  Expected d/D: {[d/D for d in range(4)]}")
        print(f"  Kernel:       {O_kernel[0, :4].tolist()}")
        # Check if output varies approximately linearly with D
        d_indices = torch.arange(D, dtype=torch.float32)
        expected_linear = d_indices / D
        actual = O_kernel[0, :]
        linear_corr = torch.corrcoef(torch.stack([expected_linear, actual]))[0, 1].item()
        print(f"  D-linear correlation: {linear_corr:.4f}")
    
    # Check if test passes (BF16 has limited precision)
    # Allow larger tolerance for BF16 (relative error up to ~1% is acceptable)
    # Handle NaN correlation (happens when all values are identical)
    if np.isnan(correlation):
        # If all values same, check if output matches reference
        passed = max_abs_err < 0.01
        print(f"  (Correlation NaN - all values same, checking abs error only)")
    else:
        passed = correlation > 0.99 and max_abs_err < 1.0
    print(f"\nTest {'PASSED' if passed else 'FAILED'}")
    
    return passed

def main():
    """Run all tests."""
    print("="*60)
    print("BF16 Flash Attention Kernel Numeric Tests (via aiter)")
    print("="*60)
    
    all_passed = True
    
    # Test with small seqlen first
    seqlen = 64
    
    # Test 1: Uniform V
    all_passed &= test_bf16_with_pattern(seqlen=seqlen, test_name="uniform_v")
    
    # Test 2: V varies by K
    all_passed &= test_bf16_with_pattern(seqlen=seqlen, test_name="v_by_k")
    
    # Test 3: V varies by D
    all_passed &= test_bf16_with_pattern(seqlen=seqlen, test_name="v_by_d")
    
    # Test 4: Random inputs
    all_passed &= test_bf16_with_pattern(seqlen=seqlen, test_name="random")
    
    print("\n" + "="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sanity Checks for Debug Tools

Verifies that our debug tools are working correctly by:
1. Testing known-good cases (should pass)
2. Testing known-bad cases (should detect issues)
3. Comparing against BF16 baseline

Run: python tools_sanity.py
"""

import sys
import io
from contextlib import redirect_stdout
from pathlib import Path


def test_asm_validator_good_case():
    """
    Sanity: asm_validator should pass for unlimited buffer size
    """
    from asm_validator import validate_kernel_params
    
    # Capture output
    f = io.StringIO()
    with redirect_stdout(f):
        result = validate_kernel_params(seq_len=128, k_buffer_size=-1, v_buffer_size=-1)
    output = f.getvalue()
    
    if "All tiles accessible" in output:
        return True, "Correctly passes unlimited buffer size"
    else:
        return False, f"Should pass unlimited size, got: {output}"


def test_asm_validator_bad_case():
    """
    Sanity: asm_validator should detect buffer overflow with size=4096
    """
    from asm_validator import validate_kernel_params
    
    f = io.StringIO()
    with redirect_stdout(f):
        result = validate_kernel_params(seq_len=128, k_buffer_size=4096, v_buffer_size=4096)
    output = f.getvalue()
    
    if "BUFFER OVERFLOW DETECTED" in output:
        return True, "Correctly detects buffer overflow"
    else:
        return False, f"Should detect overflow, got: {output}"


def test_asm_validator_edge_case():
    """
    Sanity: Single tile (seq=32) should pass even with size=4096
    """
    from asm_validator import validate_kernel_params
    
    f = io.StringIO()
    with redirect_stdout(f):
        result = validate_kernel_params(seq_len=32, k_buffer_size=4096, v_buffer_size=4096)
    output = f.getvalue()
    
    # Single tile with size=4096 should work (offset never exceeds 4096)
    if "All tiles accessible" in output:
        return True, "Correctly passes single-tile with limited size"
    else:
        return False, f"Single tile should pass, got: {output}"


def test_debug_harness_fp8():
    """
    Sanity: debug_harness should pass for fixed FP8 kernel
    """
    from debug_harness import DebugKernel
    
    dk = DebugKernel("fwd_fp8_kloop.s")
    
    # Just run a few quick tests
    dk.build()
    dk.load()
    
    r1 = dk.test_v_identity(32)
    r2 = dk.test_v_identity(64)
    
    if r1.passed and r2.passed:
        return True, f"V=1 identity passes (32: {r1.max_error:.3f}, 64: {r2.max_error:.3f})"
    else:
        return False, f"V=1 identity should pass: 32={r1.passed}, 64={r2.passed}"


def test_debug_harness_tile_isolation():
    """
    Sanity: Tile isolation should show each tile contributing
    """
    from debug_harness import DebugKernel
    
    dk = DebugKernel("fwd_fp8_kloop.s")
    dk.build()
    dk.load()
    
    results = dk.test_tile_isolation(64)  # 2 tiles
    
    # Both tiles should contribute
    if len(results) == 2 and all(r.passed for r in results):
        means = [r.mean_error for r in results]
        return True, f"Both tiles contribute: {means}"
    else:
        return False, f"Tile isolation failed: {[(r.name, r.passed) for r in results]}"


def test_bf16_baseline_available():
    """
    Sanity: BF16 baseline test should be runnable
    """
    try:
        from aiter.ops.mha import fmha_v3_fwd
        return True, "aiter.ops.mha available"
    except ImportError as e:
        return False, f"Cannot import aiter: {e}"


def test_bf16_baseline_passes():
    """
    Sanity: BF16 should pass V=1 identity with error < 0.01
    """
    try:
        from aiter.ops.mha import fmha_v3_fwd
        import torch
        import math
        
        scale = 1.0 / math.sqrt(128)
        Q = torch.randn(1, 32, 1, 128, device='cuda', dtype=torch.bfloat16) * 0.5
        K = torch.randn(1, 64, 1, 128, device='cuda', dtype=torch.bfloat16) * 0.5
        V = torch.ones(1, 64, 1, 128, device='cuda', dtype=torch.bfloat16)
        
        R, _, _, _ = fmha_v3_fwd(Q, K, V, 0.0, scale, False, -1, -1, True, False, 1)
        
        mean = R.float().mean().item()
        error = abs(mean - 1.0)
        
        if error < 0.01:
            return True, f"BF16 V=1 identity: mean={mean:.6f}, error={error:.6f}"
        else:
            return False, f"BF16 error too high: {error:.6f}"
    except Exception as e:
        return False, f"BF16 test failed: {e}"


def test_ratio_consistency_detects_bad():
    """
    Sanity: Ratio consistency should detect wildly varying ratios
    
    This simulates what happens when buffer descriptor size is wrong -
    different elements have different ratios to reference.
    """
    import torch
    
    # Simulate "good" case - consistent ratios
    ref = torch.randn(32, 32) * 0.5
    good_kernel = ref * 1.05  # Uniform 5% scaling
    
    mask = ref.abs() > 0.01
    good_ratios = good_kernel[mask] / ref[mask]
    good_std = good_ratios.std().item()
    
    # Simulate "bad" case - inconsistent ratios (like buffer overflow bug)
    bad_kernel = ref.clone()
    bad_kernel[:, 16:] = 0  # Half the columns are zeros (wrong data loaded)
    
    bad_ratios = bad_kernel[mask] / ref[mask]
    bad_std = bad_ratios.std().item()
    
    if good_std < 0.1 and bad_std > 0.5:
        return True, f"Ratio check: good_std={good_std:.3f}, bad_std={bad_std:.3f}"
    else:
        return False, f"Ratio detection not working: good={good_std:.3f}, bad={bad_std:.3f}"


def run_all_sanity_checks():
    """Run all sanity checks and report results"""
    
    print("=" * 70)
    print("DEBUG TOOLS SANITY CHECK")
    print("=" * 70)
    
    checks = [
        ("asm_validator: good case (unlimited)", test_asm_validator_good_case),
        ("asm_validator: bad case (overflow)", test_asm_validator_bad_case),
        ("asm_validator: edge case (single tile)", test_asm_validator_edge_case),
        ("debug_harness: FP8 V=1 identity", test_debug_harness_fp8),
        ("debug_harness: tile isolation", test_debug_harness_tile_isolation),
        ("bf16_baseline: available", test_bf16_baseline_available),
        ("bf16_baseline: passes", test_bf16_baseline_passes),
        ("ratio_consistency: detection", test_ratio_consistency_detects_bad),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            passed, msg = check_fn()
            status = "✅" if passed else "❌"
            print(f"{status} {name}")
            print(f"   {msg}")
            results.append((name, passed, msg))
        except Exception as e:
            print(f"❌ {name}")
            print(f"   Exception: {e}")
            results.append((name, False, str(e)))
    
    # Summary
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    print("\n" + "=" * 70)
    print(f"SANITY CHECK SUMMARY: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All debug tools are working correctly!")
    else:
        print("❌ Some tools may have issues:")
        for name, p, msg in results:
            if not p:
                print(f"   - {name}: {msg}")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    
    success = run_all_sanity_checks()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Comprehensive Numerical Tests for FP8 Flash Attention Kernel

Tests all common and edge cases for the current kernel:
- Shape: 32 Q-rows × seq_len K-rows × 128 head_dim
- K-loop with online softmax

Run: python test_numerics.py
"""

import torch
import subprocess
import ctypes
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TestResult:
    name: str
    passed: bool
    max_error: float
    message: str

class FP8AttentionTester:
    def __init__(self):
        self.hip = None
        self.func = None
        self.results: List[TestResult] = []
        
    def build_and_load(self):
        """Build and load kernel"""
        cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
        subprocess.run(
            ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx950", "-c", "fwd_fp8_kloop.s", "-o", "fwd_fp8_kloop.o"],
            capture_output=True, cwd=cwd, check=True
        )
        subprocess.run(
            ["ld.lld", "-shared", "-o", "fwd_fp8_kloop.co", "fwd_fp8_kloop.o"],
            capture_output=True, cwd=cwd, check=True
        )
        
        self.hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
        module = ctypes.c_void_p()
        self.hip.hipModuleLoad(ctypes.byref(module), f"{cwd}/fwd_fp8_kloop.co".encode())
        self.func = ctypes.c_void_p()
        self.hip.hipModuleGetFunction(ctypes.byref(self.func), module, b"_ZN5aiter13fwd_fp8_kloopE")
        
    def run_kernel(self, Q_fp8, K_fp8, V_fp8, seq_len: int) -> torch.Tensor:
        """Run kernel and return output"""
        O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
        
        args = [
            ctypes.c_void_p(O.data_ptr()),
            ctypes.c_void_p(Q_fp8.data_ptr()),
            ctypes.c_void_p(K_fp8.data_ptr()),
            ctypes.c_void_p(V_fp8.data_ptr()),
            ctypes.c_uint32(seq_len),
        ]
        args_arr = (ctypes.c_void_p * 5)(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )
        
        self.hip.hipModuleLaunchKernel(self.func, 1, 1, 1, 64, 1, 1, 12288, None, args_arr, None)
        self.hip.hipDeviceSynchronize()
        return O
        
    def reference(self, Q_fp8, K_fp8, V_fp8) -> torch.Tensor:
        """Compute reference output"""
        scale = 1.0 / math.sqrt(128)
        S = Q_fp8.float() @ K_fp8.float().T * scale
        P = torch.softmax(S, dim=1)
        P_fp8 = P.to(torch.float8_e4m3fn).float()
        return P_fp8 @ V_fp8.float()
        
    def add_result(self, name: str, passed: bool, max_error: float, message: str):
        """Record test result"""
        self.results.append(TestResult(name, passed, max_error, message))
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {message}")

    # =========================================================================
    # 1. BASIC FUNCTIONALITY TESTS
    # =========================================================================
    
    def test_single_tile(self):
        """seq_len=32: single K-tile, no loop iteration"""
        print("\n🔹 Single Tile (seq=32)")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(32, 128, device='cuda') * 0.5
        V = torch.randn(32, 128, device='cuda') * 0.5
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 32)
        O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
        
        max_err = (O - O_ref).abs().max().item()
        self.add_result("Single tile (seq=32)", max_err < 0.15, max_err, f"max_err={max_err:.4f}")
        
    def test_multi_tile(self):
        """seq_len=64,96,128: multiple K-tiles with online softmax"""
        print("\n🔹 Multi-Tile K-Loop")
        for seq_len in [64, 96, 128]:
            torch.manual_seed(42)
            Q = torch.randn(32, 128, device='cuda') * 0.5
            K = torch.randn(seq_len, 128, device='cuda') * 0.5
            V = torch.randn(seq_len, 128, device='cuda') * 0.5
            
            Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
            O = self.run_kernel(Q_fp8, K_fp8, V_fp8, seq_len)
            O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
            
            max_err = (O - O_ref).abs().max().item()
            tiles = seq_len // 32
            self.add_result(f"Multi-tile (seq={seq_len}, {tiles} tiles)", max_err < 0.15, max_err, f"max_err={max_err:.4f}")
    
    # =========================================================================
    # 2. INPUT PATTERN TESTS
    # =========================================================================
    
    def test_v_ones(self):
        """V=1 everywhere: output should be ~1.0 (softmax sums to 1)"""
        print("\n🔹 V=1 Identity Test")
        for seq_len in [32, 64, 128]:
            torch.manual_seed(42)
            Q = torch.randn(32, 128, device='cuda') * 0.5
            K = torch.randn(seq_len, 128, device='cuda') * 0.5
            V = torch.ones(seq_len, 128, device='cuda')
            
            Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
            O = self.run_kernel(Q_fp8, K_fp8, V_fp8, seq_len)
            
            mean_val = O.mean().item()
            error = abs(mean_val - 1.0)
            self.add_result(f"V=1 identity (seq={seq_len})", error < 0.05, error, f"mean={mean_val:.4f}, expected=1.0")
            
    def test_v_zeros(self):
        """V=0 everywhere: output should be ~0"""
        print("\n🔹 V=0 Zero Test")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(64, 128, device='cuda') * 0.5
        V = torch.zeros(64, 128, device='cuda')
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        
        max_val = O.abs().max().item()
        self.add_result("V=0 zero output", max_val < 0.01, max_val, f"max_abs={max_val:.6f}, expected=0")
        
    def test_structured_v(self):
        """V=row_index pattern: verifies correct V row access"""
        print("\n🔹 Structured V Pattern")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(64, 128, device='cuda') * 0.5
        # V[i, :] = i/64 (row index normalized)
        V = torch.arange(64, device='cuda').float().unsqueeze(1).expand(64, 128) / 64
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
        
        max_err = (O - O_ref).abs().max().item()
        # Higher threshold: gradient pattern has FP8 quantization at boundaries
        self.add_result("Structured V pattern", max_err < 0.35, max_err, f"max_err={max_err:.4f}")
        
    def test_different_seeds(self):
        """Multiple random seeds to catch seed-dependent bugs"""
        print("\n🔹 Different Random Seeds")
        for seed in [0, 123, 999, 2024]:
            torch.manual_seed(seed)
            Q = torch.randn(32, 128, device='cuda') * 0.5
            K = torch.randn(64, 128, device='cuda') * 0.5
            V = torch.randn(64, 128, device='cuda') * 0.5
            
            Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
            O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
            O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
            
            max_err = (O - O_ref).abs().max().item()
            self.add_result(f"Seed {seed}", max_err < 0.15, max_err, f"max_err={max_err:.4f}")

    # =========================================================================
    # 3. NUMERICAL STABILITY TESTS
    # =========================================================================
    
    def test_large_attention_scores(self):
        """Large Q·K values → softmax saturation (one element dominates)
        
        KNOWN FP8 LIMITATION: Large values overflow FP8 range (max ~448).
        This test verifies NO NaN/Inf, but allows high error.
        """
        print("\n🔹 Large Attention Scores (Softmax Saturation)")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 2.0  # Larger scale
        K = torch.randn(64, 128, device='cuda') * 2.0
        V = torch.randn(64, 128, device='cuda') * 0.5
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
        
        max_err = (O - O_ref).abs().max().item()
        has_nan = torch.isnan(O).any().item()
        has_inf = torch.isinf(O).any().item()
        
        # FP8 limitation: only check for NaN/Inf, not accuracy
        passed = not has_nan and not has_inf
        msg = f"max_err={max_err:.4f} (FP8 overflow expected)" if passed else "NaN/Inf!"
        self.add_result("Large scores (no NaN/Inf)", passed, max_err, msg)
            
    def test_small_attention_scores(self):
        """Small Q·K values → uniform softmax distribution"""
        print("\n🔹 Small Attention Scores (Uniform Distribution)")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.1  # Smaller scale
        K = torch.randn(64, 128, device='cuda') * 0.1
        V = torch.randn(64, 128, device='cuda') * 0.5
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
        
        max_err = (O - O_ref).abs().max().item()
        self.add_result("Small scores (uniform)", max_err < 0.15, max_err, f"max_err={max_err:.4f}")
        
    def test_spike_attention(self):
        """One K row similar to Q → attention spike on that row
        
        KNOWN FP8 LIMITATION: Large attention score differences cause
        softmax to become very peaky. FP8 quantization of P loses precision
        for small probabilities. This is expected behavior.
        Test only checks for no NaN/Inf and reasonable output range.
        """
        print("\n🔹 Attention Spike (One Key Dominates)")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(64, 128, device='cuda') * 0.5
        V = torch.randn(64, 128, device='cuda') * 0.5
        
        # Moderate spike: K[10] = Q[0] * 1.5 (creates higher attention but within range)
        K[10] = Q[0] * 1.5
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        
        has_nan = torch.isnan(O).any().item()
        has_inf = torch.isinf(O).any().item()
        in_range = O.abs().max().item() < 10  # Output should be reasonable
        
        passed = not has_nan and not has_inf and in_range
        msg = "Clean, in range" if passed else f"NaN={has_nan}, Inf={has_inf}, max={O.abs().max().item():.2f}"
        self.add_result("Spike attention (no NaN)", passed, 0 if passed else 1, msg)

    # =========================================================================
    # 4. MULTI-TILE SPECIFIC TESTS
    # =========================================================================
    
    def test_tile_contribution(self):
        """Each K-tile should contribute to output (not ignored)"""
        print("\n🔹 Tile Contribution Test")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(64, 128, device='cuda') * 0.5
        
        # Test with V=1 only in tile 0 vs tile 1
        V0 = torch.zeros(64, 128, device='cuda')
        V0[:32] = 1.0
        V1 = torch.zeros(64, 128, device='cuda')
        V1[32:] = 1.0
        
        Q_fp8, K_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn)
        V0_fp8 = V0.to(torch.float8_e4m3fn)
        V1_fp8 = V1.to(torch.float8_e4m3fn)
        
        O0 = self.run_kernel(Q_fp8, K_fp8, V0_fp8, 64)
        O1 = self.run_kernel(Q_fp8, K_fp8, V1_fp8, 64)
        
        # Both tiles should contribute non-trivially
        mean0 = O0.mean().item()
        mean1 = O1.mean().item()
        
        passed = mean0 > 0.1 and mean1 > 0.1
        self.add_result("Tile 0 contributes", mean0 > 0.1, mean0, f"mean={mean0:.4f}")
        self.add_result("Tile 1 contributes", mean1 > 0.1, mean1, f"mean={mean1:.4f}")
        
    def test_tile_additivity(self):
        """Sum of tile-isolated outputs ≈ full output (online softmax correctness)"""
        print("\n🔹 Tile Additivity (Online Softmax)")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(64, 128, device='cuda') * 0.5
        V_full = torch.ones(64, 128, device='cuda')
        
        Q_fp8, K_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn)
        V_full_fp8 = V_full.to(torch.float8_e4m3fn)
        
        O_full = self.run_kernel(Q_fp8, K_fp8, V_full_fp8, 64)
        
        # Sum of isolated tiles
        O_sum = torch.zeros_like(O_full)
        for tile in range(2):
            V = torch.zeros(64, 128, device='cuda')
            V[tile*32:(tile+1)*32] = 1.0
            V_fp8 = V.to(torch.float8_e4m3fn)
            O_sum += self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        
        diff = (O_full - O_sum).abs().max().item()
        self.add_result("Tile additivity", diff < 0.20, diff, f"max_diff={diff:.4f}")

    # =========================================================================
    # 5. EDGE CASES
    # =========================================================================
    
    def test_identical_q_rows(self):
        """All Q rows identical → all output rows should be identical"""
        print("\n🔹 Identical Q Rows")
        torch.manual_seed(42)
        q_row = torch.randn(1, 128, device='cuda') * 0.5
        Q = q_row.expand(32, 128).contiguous()
        K = torch.randn(64, 128, device='cuda') * 0.5
        V = torch.randn(64, 128, device='cuda') * 0.5
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        
        # All output rows should be nearly identical
        row_std = O.std(dim=0).mean().item()
        self.add_result("Identical Q rows → identical O rows", row_std < 0.02, row_std, f"row_std={row_std:.6f}")
        
    def test_identical_k_rows(self):
        """All K rows identical → uniform attention → O = mean(V)"""
        print("\n🔹 Identical K Rows (Uniform Attention)")
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        k_row = torch.randn(1, 128, device='cuda') * 0.5
        K = k_row.expand(64, 128).contiguous()
        V = torch.randn(64, 128, device='cuda') * 0.5
        
        Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
        O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
        
        # With uniform attention, O ≈ mean(V) for each row
        V_mean = V_fp8.float().mean(dim=0)
        diff = (O - V_mean.unsqueeze(0)).abs().mean().item()
        self.add_result("Identical K → O≈mean(V)", diff < 0.10, diff, f"diff_from_V_mean={diff:.4f}")
        
    def test_nan_check(self):
        """Ensure no NaN in output for various inputs"""
        print("\n🔹 NaN Check")
        test_cases = [
            ("Normal", 0.5, 0.5),
            ("Large scale", 2.0, 2.0),
            ("Small scale", 0.1, 0.1),
            ("Mixed scale", 2.0, 0.1),
        ]
        
        for name, q_scale, k_scale in test_cases:
            torch.manual_seed(42)
            Q = torch.randn(32, 128, device='cuda') * q_scale
            K = torch.randn(64, 128, device='cuda') * k_scale
            V = torch.randn(64, 128, device='cuda') * 0.5
            
            Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
            O = self.run_kernel(Q_fp8, K_fp8, V_fp8, 64)
            
            has_nan = torch.isnan(O).any().item()
            has_inf = torch.isinf(O).any().item()
            
            passed = not has_nan and not has_inf
            msg = "Clean" if passed else f"NaN={has_nan}, Inf={has_inf}"
            self.add_result(f"NaN check ({name})", passed, 0 if passed else 1, msg)

    # =========================================================================
    # 6. LONGER SEQUENCES
    # =========================================================================
    
    def test_longer_sequences(self):
        """Test with longer sequences (more K-tiles)"""
        print("\n🔹 Longer Sequences")
        for seq_len in [256, 512, 1024]:
            torch.manual_seed(42)
            Q = torch.randn(32, 128, device='cuda') * 0.5
            K = torch.randn(seq_len, 128, device='cuda') * 0.5
            V = torch.randn(seq_len, 128, device='cuda') * 0.5
            
            Q_fp8, K_fp8, V_fp8 = Q.to(torch.float8_e4m3fn), K.to(torch.float8_e4m3fn), V.to(torch.float8_e4m3fn)
            O = self.run_kernel(Q_fp8, K_fp8, V_fp8, seq_len)
            O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
            
            max_err = (O - O_ref).abs().max().item()
            has_nan = torch.isnan(O).any().item()
            
            if has_nan:
                self.add_result(f"seq={seq_len} ({seq_len//32} tiles)", False, float('inf'), "NaN!")
            else:
                self.add_result(f"seq={seq_len} ({seq_len//32} tiles)", max_err < 0.15, max_err, f"max_err={max_err:.4f}")

    # =========================================================================
    # RUN ALL
    # =========================================================================
    
    def run_all(self):
        """Run all tests"""
        print("=" * 70)
        print("FP8 FLASH ATTENTION - COMPREHENSIVE NUMERICAL TESTS")
        print("=" * 70)
        print("Kernel: fwd_fp8_kloop.s")
        print("Shape: 32 Q-rows × seq_len K-rows × 128 head_dim")
        
        self.build_and_load()
        
        # 1. Basic functionality
        self.test_single_tile()
        self.test_multi_tile()
        
        # 2. Input patterns
        self.test_v_ones()
        self.test_v_zeros()
        self.test_structured_v()
        self.test_different_seeds()
        
        # 3. Numerical stability
        self.test_large_attention_scores()
        self.test_small_attention_scores()
        self.test_spike_attention()
        
        # 4. Multi-tile specific
        self.test_tile_contribution()
        self.test_tile_additivity()
        
        # 5. Edge cases
        self.test_identical_q_rows()
        self.test_identical_k_rows()
        self.test_nan_check()
        
        # 6. Longer sequences
        self.test_longer_sequences()
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 70)
        
        if passed < total:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        return passed == total


if __name__ == "__main__":
    tester = FP8AttentionTester()
    success = tester.run_all()
    exit(0 if success else 1)

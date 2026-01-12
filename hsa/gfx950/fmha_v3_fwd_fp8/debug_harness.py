#!/usr/bin/env python3
"""
Debug Harness for FP8 Flash Attention Kernels

Provides instrumented testing with:
1. Pre-launch validation
2. Structured input patterns for debugging
3. Output anomaly detection
4. Tile-by-tile verification

Usage:
    from debug_harness import DebugKernel
    
    dk = DebugKernel("fwd_fp8_kloop.s")
    dk.test_tile_isolation()    # Test each tile independently
    dk.test_accumulation()      # Test multi-tile accumulation
    dk.test_structured_v()      # Test with V = row_index pattern
"""

import torch
import subprocess
import ctypes
import math
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    passed: bool
    max_error: float
    mean_error: float
    message: str = ""
    details: Dict = None


class DebugKernel:
    """Instrumented kernel wrapper for debugging"""
    
    def __init__(self, asm_file: str, build_dir: str = None):
        self.asm_path = Path(asm_file)
        self.build_dir = Path(build_dir) if build_dir else self.asm_path.parent
        self.co_path = self.build_dir / self.asm_path.stem.replace('.s', '.co')
        
        self.hip = None
        self.module = None
        self.func = None
        self.func_name = "_ZN5aiter13fwd_fp8_kloopE"  # Default
        
        self.results: List[TestResult] = []
        
    def build(self):
        """Build the kernel"""
        o_path = self.build_dir / (self.asm_path.stem + '.o')
        
        result = subprocess.run([
            'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
            '-mcpu=gfx950', '-c', str(self.asm_path), '-o', str(o_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Compile failed: {result.stderr}")
            
        result = subprocess.run([
            'ld.lld', '-shared', '-o', str(self.co_path), str(o_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Link failed: {result.stderr}")
            
    def load(self):
        """Load the kernel module"""
        if not self.co_path.exists():
            self.build()
            
        self.hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
        self.module = ctypes.c_void_p()
        self.hip.hipModuleLoad(ctypes.byref(self.module), str(self.co_path).encode())
        self.func = ctypes.c_void_p()
        self.hip.hipModuleGetFunction(ctypes.byref(self.func), self.module, self.func_name.encode())
        
    def launch(self, O, Q_fp8, K_fp8, V_fp8, seq_len: int, shared_mem: int = 12288):
        """Launch the kernel"""
        if self.func is None:
            self.load()
            
        args = [ctypes.c_void_p(x.data_ptr()) for x in [O, Q_fp8, K_fp8, V_fp8]]
        args.append(ctypes.c_uint32(seq_len))
        args_arr = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        
        self.hip.hipModuleLaunchKernel(
            self.func, 1, 1, 1, 64, 1, 1, shared_mem, None, args_arr, None
        )
        self.hip.hipDeviceSynchronize()
        
    def reference(self, Q_fp8, K_fp8, V_fp8) -> torch.Tensor:
        """Compute reference output"""
        scale = 1.0 / math.sqrt(128)
        S = Q_fp8.float() @ K_fp8.float().T * scale
        P = torch.softmax(S, dim=1)
        P_fp8 = P.to(torch.float8_e4m3fn).float()
        return P_fp8 @ V_fp8.float()
        
    # =========================================================================
    # Structured Tests
    # =========================================================================
    
    def test_v_identity(self, seq_len: int = 64) -> TestResult:
        """
        Test with V = 1.0 everywhere.
        Expected: O = 1.0 (sum of softmax = 1)
        """
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(seq_len, 128, device='cuda') * 0.5
        V = torch.ones(seq_len, 128, device='cuda')
        
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        V_fp8 = V.to(torch.float8_e4m3fn)
        O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
        
        self.launch(O, Q_fp8, K_fp8, V_fp8, seq_len)
        
        expected = 1.0
        actual_mean = O[:, :32].mean().item()
        error = abs(actual_mean - expected)
        
        passed = error < 0.1
        result = TestResult(
            name=f"V=1 identity (seq={seq_len})",
            passed=passed,
            max_error=error,
            mean_error=error,
            message=f"Expected mean=1.0, got {actual_mean:.4f}",
            details={'expected': expected, 'actual_mean': actual_mean}
        )
        self.results.append(result)
        return result
        
    def test_tile_isolation(self, seq_len: int = 64) -> List[TestResult]:
        """
        Test each tile independently by setting V=0 except for one tile.
        This verifies that each tile's data is being accessed correctly.
        """
        results = []
        n_tiles = seq_len // 32
        
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(seq_len, 128, device='cuda') * 0.5
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        
        for test_tile in range(n_tiles):
            # V is 1.0 only for test_tile, 0 elsewhere
            V = torch.zeros(seq_len, 128, device='cuda')
            V[test_tile*32:(test_tile+1)*32, :] = 1.0
            V_fp8 = V.to(torch.float8_e4m3fn)
            
            O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
            self.launch(O, Q_fp8, K_fp8, V_fp8, seq_len)
            
            O_mean = O[:, :32].mean().item()
            
            # Expected: O should be proportional to P_sum for this tile
            # Roughly 1/n_tiles if attention is uniform
            expected_range = (0.1, 0.9)  # Reasonable range
            passed = O_mean > 0.01  # Should be non-zero!
            
            result = TestResult(
                name=f"Tile {test_tile} isolation (seq={seq_len})",
                passed=passed,
                max_error=0 if passed else 1.0,
                mean_error=O_mean,
                message=f"O mean={O_mean:.4f} (should be >0.01 if tile {test_tile} used)",
                details={'tile': test_tile, 'o_mean': O_mean}
            )
            results.append(result)
            self.results.append(result)
            
        return results
        
    def test_accumulation(self, seq_len: int = 64) -> TestResult:
        """
        Test that tiles accumulate correctly.
        Sum of tile-isolated outputs should equal full output.
        """
        n_tiles = seq_len // 32
        
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(seq_len, 128, device='cuda') * 0.5
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        
        # Full V=1 test
        V_full = torch.ones(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        O_full = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
        self.launch(O_full, Q_fp8, K_fp8, V_full, seq_len)
        
        # Sum of isolated tiles
        O_sum = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
        for tile in range(n_tiles):
            V = torch.zeros(seq_len, 128, device='cuda')
            V[tile*32:(tile+1)*32, :] = 1.0
            V_fp8 = V.to(torch.float8_e4m3fn)
            
            O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
            self.launch(O, Q_fp8, K_fp8, V_fp8, seq_len)
            O_sum += O
            
        # Compare
        error = (O_full[:, :32] - O_sum[:, :32]).abs().max().item()
        passed = error < 0.2  # Allow some FP8 quantization error
        
        result = TestResult(
            name=f"Accumulation check (seq={seq_len})",
            passed=passed,
            max_error=error,
            mean_error=(O_full[:, :32] - O_sum[:, :32]).abs().mean().item(),
            message=f"full_O vs sum_of_tiles: max_diff={error:.4f}",
            details={'o_full_mean': O_full[:, :32].mean().item(), 
                    'o_sum_mean': O_sum[:, :32].mean().item()}
        )
        self.results.append(result)
        return result
        
    def test_reference_match(self, seq_len: int = 64, threshold: float = 0.15) -> TestResult:
        """Standard reference comparison test"""
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(seq_len, 128, device='cuda') * 0.5
        V = torch.randn(seq_len, 128, device='cuda') * 0.5
        
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        V_fp8 = V.to(torch.float8_e4m3fn)
        O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
        
        self.launch(O, Q_fp8, K_fp8, V_fp8, seq_len)
        O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
        
        max_err = (O[:, :32] - O_ref[:, :32]).abs().max().item()
        mean_err = (O[:, :32] - O_ref[:, :32]).abs().mean().item()
        passed = max_err < threshold
        
        result = TestResult(
            name=f"Reference match (seq={seq_len})",
            passed=passed,
            max_error=max_err,
            mean_error=mean_err,
            message=f"max_err={max_err:.4f} vs threshold={threshold}"
        )
        self.results.append(result)
        return result
        
    def test_element_ratios(self, seq_len: int = 64) -> TestResult:
        """
        Check if output elements have consistent ratios to reference.
        Inconsistent ratios indicate data layout/addressing issues.
        """
        torch.manual_seed(42)
        Q = torch.randn(32, 128, device='cuda') * 0.5
        K = torch.randn(seq_len, 128, device='cuda') * 0.5
        V = torch.randn(seq_len, 128, device='cuda') * 0.5
        
        Q_fp8 = Q.to(torch.float8_e4m3fn)
        K_fp8 = K.to(torch.float8_e4m3fn)
        V_fp8 = V.to(torch.float8_e4m3fn)
        O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
        
        self.launch(O, Q_fp8, K_fp8, V_fp8, seq_len)
        O_ref = self.reference(Q_fp8, K_fp8, V_fp8)
        
        # Calculate ratios for elements with significant magnitude
        mask = O_ref[:, :32].abs() > 0.01
        ratios = O[:, :32][mask] / O_ref[:, :32][mask]
        
        ratio_std = ratios.std().item()
        ratio_mean = ratios.mean().item()
        
        # Good: all ratios ~1.0 with low variance
        # Bad: wildly different ratios = addressing bug
        passed = ratio_std < 0.3 and 0.7 < ratio_mean < 1.3
        
        result = TestResult(
            name=f"Element ratio consistency (seq={seq_len})",
            passed=passed,
            max_error=ratio_std,
            mean_error=abs(ratio_mean - 1.0),
            message=f"ratio_mean={ratio_mean:.3f}, ratio_std={ratio_std:.3f}",
            details={'ratio_mean': ratio_mean, 'ratio_std': ratio_std,
                    'sample_ratios': ratios[:10].tolist()}
        )
        self.results.append(result)
        return result
        
    # =========================================================================
    # Run all tests
    # =========================================================================
    
    def run_all(self, seq_lens: List[int] = [32, 64, 128]):
        """Run comprehensive test suite"""
        print("=" * 70)
        print(f"DEBUG HARNESS: {self.asm_path.name}")
        print("=" * 70)
        
        self.build()
        self.load()
        
        for seq_len in seq_lens:
            print(f"\n📋 Testing seq_len={seq_len} ({seq_len//32} tiles)")
            print("-" * 50)
            
            # Run tests
            r = self.test_v_identity(seq_len)
            print(f"  {'✅' if r.passed else '❌'} {r.name}: {r.message}")
            
            tile_results = self.test_tile_isolation(seq_len)
            for r in tile_results:
                print(f"  {'✅' if r.passed else '❌'} {r.name}: {r.message}")
                
            r = self.test_accumulation(seq_len)
            print(f"  {'✅' if r.passed else '❌'} {r.name}: {r.message}")
            
            r = self.test_reference_match(seq_len)
            print(f"  {'✅' if r.passed else '❌'} {r.name}: {r.message}")
            
            r = self.test_element_ratios(seq_len)
            print(f"  {'✅' if r.passed else '❌'} {r.name}: {r.message}")
            
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 70)
        
        return all(r.passed for r in self.results)


if __name__ == "__main__":
    import sys
    asm_file = sys.argv[1] if len(sys.argv) > 1 else "fwd_fp8_kloop.s"
    
    dk = DebugKernel(asm_file)
    dk.run_all([32, 64, 128])

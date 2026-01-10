#!/usr/bin/env python3
"""Rigorous tests for integrate_step4_hd128.s - FP8 Flash Attention head_dim=128"""

import torch
import subprocess
import ctypes
import numpy as np
import sys

def build_kernel():
    src = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.s"
    obj = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.o"
    co = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.co"
    
    cmd = f"/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c {src} -o {obj}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Assembly failed:", result.stderr)
        return False
    
    cmd = f"/opt/rocm/llvm/bin/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx950 {obj} -o {co}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Linking failed:", result.stderr)
        return False
    return True

def load_kernel():
    co_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.co"
    hip = ctypes.CDLL("libamdhip64.so")
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    
    function = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(function), module, b"_ZN5aiter21integrate_step4_hd128E")
    
    return hip, module, function

def launch_kernel(hip, function, O, Q, K, V):
    args_array = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(O.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(Q.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(K.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(V.data_ptr())), ctypes.c_void_p),
    )
    
    status = hip.hipModuleLaunchKernel(
        function, 1, 1, 1, 64, 1, 1, 6144, None, args_array, None
    )
    if status != 0:
        raise RuntimeError(f"Launch failed: {status}")
    hip.hipDeviceSynchronize()

def reference_attention(Q, K, V):
    """PyTorch reference"""
    Q_f32 = Q.view(torch.float8_e4m3fn).to(torch.float32)
    K_f32 = K.view(torch.float8_e4m3fn).to(torch.float32)
    V_f32 = V.view(torch.float8_e4m3fn).to(torch.float32)
    
    S = torch.matmul(Q_f32, K_f32.T)
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V_f32)
    return O

def run_single_test(hip, function, name, Q, K, V, verbose=True):
    """Run one test and return results"""
    O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    launch_kernel(hip, function, O, Q, K, V)
    O_ref = reference_attention(Q, K, V)
    
    O_cpu = O.cpu().numpy()
    O_ref_cpu = O_ref.cpu().numpy()
    
    diff = np.abs(O_cpu - O_ref_cpu)
    max_err = diff.max()
    mean_err = diff.mean()
    
    # Per-row analysis
    row_max_errs = diff.max(axis=1)
    row_mean_errs = diff.mean(axis=1)
    
    # Correlation
    corr = np.corrcoef(O_cpu.flatten(), O_ref_cpu.flatten())[0, 1]
    if np.isnan(corr):
        corr = 1.0 if max_err < 0.001 else 0.0
    
    passed = max_err < 0.1 and corr > 0.99
    
    if verbose:
        print(f"\n{name}:")
        print(f"  Max err: {max_err:.6f}, Mean err: {mean_err:.6f}, Corr: {corr:.6f}")
        print(f"  Row max errs - min: {row_max_errs.min():.6f}, max: {row_max_errs.max():.6f}")
        if not passed:
            # Find worst row
            worst_row = np.argmax(row_max_errs)
            worst_col = np.argmax(diff[worst_row])
            print(f"  Worst: row={worst_row}, col={worst_col}")
            print(f"    Kernel: {O_cpu[worst_row, max(0,worst_col-2):worst_col+3]}")
            print(f"    Ref:    {O_ref_cpu[worst_row, max(0,worst_col-2):worst_col+3]}")
        print(f"  {'PASS' if passed else 'FAIL'}")
    
    return passed, max_err, mean_err, corr

def main():
    print("="*70)
    print("RIGOROUS TESTING: FP8 Flash Attention head_dim=128")
    print("="*70)
    
    print("\nBuilding kernel...")
    if not build_kernel():
        return 1
    
    hip, module, function = load_kernel()
    
    results = []
    
    # ==================== Category 1: Uniform values ====================
    print("\n" + "="*70)
    print("CATEGORY 1: Uniform Values")
    print("="*70)
    
    for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        Q = torch.full((32, 128), val, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        K = torch.full((32, 128), val, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        V = torch.full((32, 128), val, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        p, *_ = run_single_test(hip, function, f"Uniform {val}", Q, K, V)
        results.append(("Uniform", p))
    
    # ==================== Category 2: Random with various seeds ====================
    print("\n" + "="*70)
    print("CATEGORY 2: Random Values (10 seeds)")
    print("="*70)
    
    for seed in range(10):
        torch.manual_seed(seed * 1000)
        Q = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        V = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        p, *_ = run_single_test(hip, function, f"Random seed={seed*1000}", Q, K, V)
        results.append(("Random", p))
    
    # ==================== Category 3: Different scales ====================
    print("\n" + "="*70)
    print("CATEGORY 3: Different Scales")
    print("="*70)
    
    for scale in [0.1, 0.2, 0.4, 0.6, 0.8]:
        torch.manual_seed(42)
        Q = (torch.randn(32, 128) * scale).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        K = (torch.randn(32, 128) * scale).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        V = (torch.randn(32, 128) * scale).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        p, *_ = run_single_test(hip, function, f"Scale {scale}", Q, K, V)
        results.append(("Scale", p))
    
    # ==================== Category 4: Sparse attention patterns ====================
    print("\n" + "="*70)
    print("CATEGORY 4: Sparse/Structured Patterns")
    print("="*70)
    
    # One-hot Q (each query attends to one key)
    for hot_row in [0, 15, 31]:
        Q = torch.zeros(32, 128)
        Q[hot_row, :] = 1.0
        Q = Q.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        V = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        p, *_ = run_single_test(hip, function, f"One-hot Q row={hot_row}", Q, K, V)
        results.append(("Sparse", p))
    
    # Diagonal dominance
    torch.manual_seed(123)
    Q = torch.randn(32, 128) * 0.1
    K = torch.randn(32, 128) * 0.1
    for i in range(32):
        Q[i, i*4:(i+1)*4] += 1.0
        K[i, i*4:(i+1)*4] += 1.0
    Q = Q.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = K.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    p, *_ = run_single_test(hip, function, "Diagonal dominant Q/K", Q, K, V)
    results.append(("Sparse", p))
    
    # ==================== Category 5: Edge cases ====================
    print("\n" + "="*70)
    print("CATEGORY 5: Edge Cases")
    print("="*70)
    
    # Very small values
    Q = torch.full((32, 128), 0.01, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = torch.full((32, 128), 0.01, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = torch.full((32, 128), 0.5, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    p, *_ = run_single_test(hip, function, "Small Q/K, normal V", Q, K, V)
    results.append(("Edge", p))
    
    # Mixed positive/negative
    torch.manual_seed(999)
    Q = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    p, *_ = run_single_test(hip, function, "Mixed +/- values", Q, K, V)
    results.append(("Edge", p))
    
    # All same row in Q
    Q = torch.zeros(32, 128)
    Q[:, :] = (torch.randn(128) * 0.3).unsqueeze(0)
    Q = Q.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    p, *_ = run_single_test(hip, function, "Same row Q (all queries identical)", Q, K, V)
    results.append(("Edge", p))
    
    # ==================== Category 6: D-tile boundary tests ====================
    print("\n" + "="*70)
    print("CATEGORY 6: D-tile Boundary Tests")
    print("="*70)
    
    # V with distinct values per D-tile
    torch.manual_seed(555)
    Q = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = torch.zeros(32, 128)
    V[:, 0:32] = 0.1
    V[:, 32:64] = 0.3
    V[:, 64:96] = 0.5
    V[:, 96:128] = 0.7
    V = V.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    p, *_ = run_single_test(hip, function, "V with D-tile patterns", Q, K, V)
    results.append(("Dtile", p))
    
    # Check each D-tile separately
    for d_start in [0, 32, 64, 96]:
        torch.manual_seed(777 + d_start)
        Q = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        V = torch.zeros(32, 128)
        V[:, d_start:d_start+32] = torch.randn(32, 32) * 0.5
        V = V.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        p, *_ = run_single_test(hip, function, f"V nonzero only D={d_start}:{d_start+32}", Q, K, V)
        results.append(("Dtile", p))
    
    # ==================== Category 7: Row-by-row validation ====================
    print("\n" + "="*70)
    print("CATEGORY 7: Row-by-Row Validation")
    print("="*70)
    
    torch.manual_seed(12345)
    Q = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    
    O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    launch_kernel(hip, function, O, Q, K, V)
    O_ref = reference_attention(Q, K, V)
    
    O_cpu = O.cpu().numpy()
    O_ref_cpu = O_ref.cpu().numpy()
    
    print("\nPer-row analysis:")
    row_pass = 0
    for row in range(32):
        row_diff = np.abs(O_cpu[row] - O_ref_cpu[row])
        row_max = row_diff.max()
        row_mean = row_diff.mean()
        status = "OK" if row_max < 0.1 else "ERR"
        if row_max >= 0.05 or row < 4 or row >= 28:  # Show first/last rows and any issues
            print(f"  Row {row:2d}: max={row_max:.6f} mean={row_mean:.6f} [{status}]")
        if row_max < 0.1:
            row_pass += 1
    print(f"\nRows passing: {row_pass}/32")
    results.append(("RowValidation", row_pass == 32))
    
    # ==================== Summary ====================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    categories = {}
    for cat, passed in results:
        if cat not in categories:
            categories[cat] = [0, 0]
        categories[cat][1] += 1
        if passed:
            categories[cat][0] += 1
    
    total_pass = sum(p for _, p in results)
    total_tests = len(results)
    
    print(f"\nBy category:")
    for cat, (p, t) in categories.items():
        status = "✓" if p == t else "✗"
        print(f"  {cat:15s}: {p:2d}/{t:2d} {status}")
    
    print(f"\nOverall: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("\n✓ ALL TESTS PASSED - Kernel is numerically correct!")
        return 0
    else:
        print(f"\n✗ {total_tests - total_pass} TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

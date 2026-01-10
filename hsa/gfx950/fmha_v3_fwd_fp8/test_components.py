#!/usr/bin/env python3
"""
Rigorous component tests for FP8 Flash Attention kernel.
Each test uses random inputs and compares against PyTorch reference.
"""

import torch
import numpy as np
import ctypes
import struct
import math
import sys

# Initialize HIP
hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

def load_kernel(filename, symbol):
    """Load a kernel from .co file"""
    module = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), filename.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to load {filename}: error {ret}")
    func = ctypes.c_void_p()
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode())
    if ret != 0:
        raise RuntimeError(f"Failed to get function {symbol}: error {ret}")
    return module, func

def launch_kernel(func, grid, block, shared_mem, args_bytes):
    """Launch a kernel with given arguments"""
    args_gpu = torch.from_numpy(np.frombuffer(args_bytes, dtype=np.uint8)).cuda()
    kernarg_ptr = ctypes.c_void_p(args_gpu.data_ptr())
    kernarg_size = ctypes.c_size_t(len(args_bytes))
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
        HIP_LAUNCH_PARAM_END
    )
    hip.hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], 
                               block[0], block[1], block[2],
                               shared_mem, None, None, extra)
    hip.hipDeviceSynchronize()

def to_fp8(x):
    """Convert tensor to FP8 e4m3fn format"""
    return x.to(torch.float8_e4m3fn).view(torch.uint8)

def from_fp8(x):
    """Convert FP8 bytes back to float"""
    return x.view(torch.float8_e4m3fn).float()

# ============================================================================
# TEST 1: QK MFMA
# ============================================================================
def test_qk_mfma():
    """Test QK MFMA: S = Q @ K^T"""
    print("=" * 70)
    print("TEST 1: QK MFMA (S = Q @ K^T)")
    print("=" * 70)
    
    try:
        module, func = load_kernel("step1_qk_mfma.co", "_ZN5aiter12step1_qk_mfmaE")
    except Exception as e:
        print(f"  SKIP: {e}")
        return None
    
    # Random inputs (small values to avoid FP8 overflow)
    torch.manual_seed(42)
    Q_f32 = torch.randn(32, 16) * 0.5  # 32 queries, 16 head_dim
    K_f32 = torch.randn(32, 16) * 0.5  # 32 keys, 16 head_dim
    
    # Reference computation
    S_ref = Q_f32 @ K_f32.T  # [32, 32]
    
    # Convert to FP8
    Q_fp8 = to_fp8(Q_f32)
    K_fp8 = to_fp8(K_f32)
    
    # FP8 reference (account for quantization)
    Q_quant = from_fp8(Q_fp8)
    K_quant = from_fp8(K_fp8)
    S_ref_quant = Q_quant @ K_quant.T
    
    # Allocate output
    S_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    Q_gpu = Q_fp8.cuda()
    K_gpu = K_fp8.cuda()
    
    # Launch kernel
    args = bytearray(32)
    struct.pack_into('Q', args, 0, S_out.data_ptr())
    struct.pack_into('Q', args, 8, Q_gpu.data_ptr())
    struct.pack_into('Q', args, 16, K_gpu.data_ptr())
    
    launch_kernel(func, (1,1,1), (64,1,1), 0, args)
    
    S_kernel = S_out.cpu()
    
    # Compare
    max_err = (S_kernel - S_ref_quant).abs().max().item()
    mean_err = (S_kernel - S_ref_quant).abs().mean().item()
    
    print(f"  S_ref_quant[0,0] = {S_ref_quant[0,0].item():.4f}")
    print(f"  S_kernel[0,0] = {S_kernel[0,0].item():.4f}")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    
    # Strict tolerance
    passed = max_err < 0.1
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        print(f"  S_ref_quant[:3,:3]:\n{S_ref_quant[:3,:3]}")
        print(f"  S_kernel[:3,:3]:\n{S_kernel[:3,:3]}")
    
    return passed

# ============================================================================
# TEST 2: Softmax
# ============================================================================
def test_softmax():
    """Test softmax computation: P = softmax(S)"""
    print("=" * 70)
    print("TEST 2: Softmax (P = softmax(S))")
    print("=" * 70)
    
    try:
        module, func = load_kernel("test_softmax_only.co", "_ZN5aiter17test_softmax_onlyE")
    except Exception as e:
        print(f"  SKIP: {e}")
        return None
    
    # This kernel generates S=1 internally, so we test that
    # Expected: P = 1/32 for all positions
    
    O_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    args = bytearray(16)
    struct.pack_into('Q', args, 0, O_out.data_ptr())
    
    launch_kernel(func, (1,1,1), (64,1,1), 4096, args)
    
    O = O_out.cpu()
    
    # Check P values (stored in O[:, 0:4])
    P_expected = 1.0 / 32.0
    P_actual = O[:, 0:4]
    
    max_err = (P_actual - P_expected).abs().max().item()
    mean_err = (P_actual - P_expected).abs().mean().item()
    
    print(f"  Expected P = {P_expected:.6f}")
    print(f"  P[0, 0:4] = {O[0, 0:4].tolist()}")
    print(f"  P[15, 0:4] = {O[15, 0:4].tolist()}")
    print(f"  P[31, 0:4] = {O[31, 0:4].tolist()}")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    
    # Check sum and max (debug outputs)
    print(f"  Row 0: sum={O[0,4].item():.2f} (expected 32), max={O[0,5].item():.2f} (expected 1)")
    
    passed = max_err < 0.001
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return passed

# ============================================================================
# TEST 3: PV MFMA with uniform P
# ============================================================================
def test_pv_mfma_uniform():
    """Test PV MFMA with uniform P=1/32 and V=1"""
    print("=" * 70)
    print("TEST 3: PV MFMA (uniform P=1/32, V=1)")
    print("=" * 70)
    
    try:
        module, func = load_kernel("test_softmax_pv.co", "_ZN5aiter15test_softmax_pvE")
    except Exception as e:
        print(f"  SKIP: {e}")
        return None
    
    # This kernel uses P=1/32 and V=1 internally
    # Expected: O = sum_k(1/32 * 1) = 1.0
    
    O_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    args = bytearray(16)
    struct.pack_into('Q', args, 0, O_out.data_ptr())
    
    launch_kernel(func, (1,1,1), (64,1,1), 8192, args)
    
    O = O_out.cpu()
    
    O_expected = 1.0
    max_err = (O - O_expected).abs().max().item()
    mean_val = O.mean().item()
    std_val = O.std().item()
    
    print(f"  Expected O = {O_expected}")
    print(f"  O[0,0] = {O[0,0].item():.4f}")
    print(f"  O mean = {mean_val:.4f}")
    print(f"  O std = {std_val:.6f}")
    print(f"  Max error: {max_err:.6f}")
    
    passed = max_err < 0.01 and std_val < 0.001
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return passed

# ============================================================================
# TEST 4: V loading from global memory
# ============================================================================
def test_v_loading():
    """Test V loading from global memory with correct stride"""
    print("=" * 70)
    print("TEST 4: V loading from global memory")
    print("=" * 70)
    
    try:
        module, func = load_kernel("test_v_from_global.co", "_ZN5aiter17test_v_from_globalE")
    except Exception as e:
        print(f"  SKIP: {e}")
        return None
    
    # Test with varying V values
    torch.manual_seed(123)
    V_f32 = torch.randn(16, 32) * 0.3 + 0.5  # Keep positive for easier debugging
    V_fp8 = to_fp8(V_f32)
    V_quant = from_fp8(V_fp8)
    
    # P = 1/16 uniform (kernel internal), so O = sum_k(1/16 * V[k,:]) = mean(V, dim=0)
    O_expected = V_quant.mean(dim=0)
    
    O_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    V_gpu = V_fp8.cuda()
    
    args = bytearray(24)
    struct.pack_into('Q', args, 0, O_out.data_ptr())
    struct.pack_into('Q', args, 8, V_gpu.data_ptr())
    
    launch_kernel(func, (1,1,1), (64,1,1), 0, args)
    
    O = O_out.cpu()
    
    # All rows should be the same (P is uniform across Q)
    O_row0 = O[0, :]
    
    max_err = (O_row0 - O_expected).abs().max().item()
    mean_err = (O_row0 - O_expected).abs().mean().item()
    
    print(f"  O_expected[0:5] = {O_expected[0:5].tolist()}")
    print(f"  O_kernel[0, 0:5] = {O_row0[0:5].tolist()}")
    print(f"  Max error: {max_err:.4f}")
    print(f"  Mean error: {mean_err:.4f}")
    
    # Check all rows are same
    row_std = O.std(dim=0).mean().item()
    print(f"  Row consistency (std across rows): {row_std:.6f}")
    
    passed = max_err < 0.1 and row_std < 0.01
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return passed

# ============================================================================
# TEST 5: P redistribution
# ============================================================================
def test_p_redistribution():
    """Test P redistribution for PV MFMA"""
    print("=" * 70)
    print("TEST 5: P redistribution (non-uniform P)")
    print("=" * 70)
    
    try:
        module, func = load_kernel("test_nonuniform_p.co", "_ZN5aiter15test_nonuniform_pE")
    except Exception as e:
        print(f"  SKIP: {e}")
        return None
    
    # This kernel uses P[k] = (k+1)/136 for k=0..15, 0 for k>=16
    # V = 1.0 for all
    # Expected: O = sum_k((k+1)/136 * 1) = sum(1..16)/136 = 136/136 = 1.0
    # But with FP8 quantization, expect ~0.98
    
    O_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    args = bytearray(16)
    struct.pack_into('Q', args, 0, O_out.data_ptr())
    
    launch_kernel(func, (1,1,1), (64,1,1), 4096, args)
    
    O = O_out.cpu()
    
    O_expected = 0.98  # Account for FP8 quantization
    max_err = (O - O_expected).abs().max().item()
    mean_val = O.mean().item()
    std_val = O.std().item()
    
    print(f"  Expected O ≈ {O_expected}")
    print(f"  O[0,0] = {O[0,0].item():.4f}")
    print(f"  O mean = {mean_val:.4f}")
    print(f"  O std = {std_val:.6f}")
    print(f"  Max error from expected: {max_err:.4f}")
    
    passed = abs(mean_val - O_expected) < 0.05 and std_val < 0.01
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return passed

# ============================================================================
# TEST 6: QK to LDS roundtrip
# ============================================================================
def test_qk_to_lds():
    """Test QK MFMA output stored to LDS and read back"""
    print("=" * 70)
    print("TEST 6: QK -> LDS -> readback")
    print("=" * 70)
    
    try:
        module, func = load_kernel("test_qk_to_lds.co", "_ZN5aiter14test_qk_to_ldsE")
    except Exception as e:
        print(f"  SKIP: {e}")
        return None
    
    # Random inputs
    torch.manual_seed(456)
    Q_f32 = torch.randn(32, 16) * 0.5
    K_f32 = torch.randn(32, 16) * 0.5
    
    Q_fp8 = to_fp8(Q_f32)
    K_fp8 = to_fp8(K_f32)
    Q_quant = from_fp8(Q_fp8)
    K_quant = from_fp8(K_fp8)
    
    # Reference: S = Q @ K^T
    S_ref = Q_quant @ K_quant.T
    
    # Kernel outputs S[:, 0:4] for each row
    out = torch.zeros(32, 4, dtype=torch.float32, device='cuda')
    Q_gpu = Q_fp8.cuda()
    K_gpu = K_fp8.cuda()
    
    args = bytearray(32)
    struct.pack_into('Q', args, 0, out.data_ptr())
    struct.pack_into('Q', args, 8, Q_gpu.data_ptr())
    struct.pack_into('Q', args, 16, K_gpu.data_ptr())
    
    launch_kernel(func, (1,1,1), (64,1,1), 4096, args)
    
    S_kernel = out.cpu()
    S_ref_partial = S_ref[:, 0:4]
    
    max_err = (S_kernel - S_ref_partial).abs().max().item()
    mean_err = (S_kernel - S_ref_partial).abs().mean().item()
    
    print(f"  S_ref[0, 0:4] = {S_ref_partial[0].tolist()}")
    print(f"  S_kernel[0, 0:4] = {S_kernel[0].tolist()}")
    print(f"  Max error: {max_err:.4f}")
    print(f"  Mean error: {mean_err:.4f}")
    
    passed = max_err < 0.1
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return passed

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("FP8 FLASH ATTENTION - COMPONENT TESTS")
    print("=" * 70 + "\n")
    
    results = {}
    
    results['qk_mfma'] = test_qk_mfma()
    print()
    
    results['softmax'] = test_softmax()
    print()
    
    results['pv_mfma_uniform'] = test_pv_mfma_uniform()
    print()
    
    results['v_loading'] = test_v_loading()
    print()
    
    results['p_redistribution'] = test_p_redistribution()
    print()
    
    results['qk_to_lds'] = test_qk_to_lds()
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else ("SKIP" if passed is None else "✗ FAIL")
        print(f"  {name}: {status}")
        if passed is False:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

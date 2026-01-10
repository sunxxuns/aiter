#!/usr/bin/env python3
"""
Test FP8 V×P flash attention kernel with non-uniform inputs.
"""

import torch
import numpy as np
import ctypes
import struct
import os

torch.manual_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_fp8_vxp_kernel(version=2):
    """Load the FP8 V×P flash attention kernel."""
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    
    if version == 2:
        co_path = os.path.join(SCRIPT_DIR, "fwd_hd128_fp8_vxp_v2.co")
        func_name = b"_ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E"
    else:
        co_path = os.path.join(SCRIPT_DIR, "fwd_hd128_fp8_vxp.co")
        func_name = b"_ZN5aiter22fmha_fwd_hd128_fp8_vxpE"
    
    module = ctypes.c_void_p()
    err = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if err != 0:
        raise RuntimeError(f"hipModuleLoad failed: {err}")
    
    func = ctypes.c_void_p()
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, func_name)
    if err != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {err}")
    
    return hip, func

def compute_reference(Q, K, V, softmax_scale):
    """Compute reference flash attention output."""
    QK = Q.float() @ K.float().T
    QK = QK * softmax_scale
    P = torch.softmax(QK, dim=-1)
    O = P @ V.float()
    return O, P

def test_fp8_vxp_kernel(seqlen_q=64, seqlen_k=32, head_dim=128, test_name="random"):
    """Test FP8 V×P kernel with specific input pattern."""
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, head_dim={head_dim}")
    print('='*60)
    
    # Initialize CUDA
    _ = torch.zeros(1, device='cuda')
    
    # Load kernel
    try:
        hip, func = load_fp8_vxp_kernel()
    except Exception as e:
        print(f"ERROR: Failed to load kernel: {e}")
        return False
    
    HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
    HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
    HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)
    
    softmax_scale = 1.0 / np.sqrt(head_dim)
    
    # Generate test data based on pattern
    if test_name == "uniform_v":
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        V = torch.ones(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        expected_behavior = "Output should be ~1.0 everywhere"
    elif test_name == "v_by_k":
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        v_vals = torch.arange(seqlen_k, dtype=torch.float32, device='cuda') / seqlen_k
        V = v_vals.unsqueeze(1).expand(seqlen_k, head_dim)
        expected_behavior = "Output should be weighted average of k indices"
    elif test_name == "v_by_d":
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        v_vals = torch.arange(head_dim, dtype=torch.float32, device='cuda') / head_dim
        V = v_vals.unsqueeze(0).expand(seqlen_k, head_dim)
        expected_behavior = "Output[:, d] should be ~d/head_dim (D preserved)"
    else:  # random
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        V = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        expected_behavior = "Output should match PyTorch reference"
    
    print(f"Expected: {expected_behavior}")
    
    # Compute reference
    O_ref, P = compute_reference(Q, K, V, softmax_scale)
    
    # Convert to FP8
    Q_fp8 = Q.to(torch.float8_e4m3fnuz).view(torch.uint8)
    K_fp8 = K.to(torch.float8_e4m3fnuz).view(torch.uint8)
    V_fp8 = V.to(torch.float8_e4m3fnuz).view(torch.uint8)
    
    # Allocate output
    output = torch.zeros(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
    
    # Pack kernel arguments
    args = bytearray(528)
    struct.pack_into('Q', args, 0x00, output.data_ptr())  # ptr_R
    struct.pack_into('Q', args, 0x10, Q_fp8.data_ptr())   # ptr_Q
    struct.pack_into('Q', args, 0x20, K_fp8.data_ptr())   # ptr_K
    struct.pack_into('Q', args, 0x30, V_fp8.data_ptr())   # ptr_V
    struct.pack_into('Q', args, 0x40, 0)                   # ptr_LSE
    struct.pack_into('f', args, 0x50, softmax_scale)       # softmax_scale
    struct.pack_into('I', args, 0x58, seqlen_q)            # seqlen_q
    struct.pack_into('I', args, 0x60, seqlen_k)            # seqlen_k
    struct.pack_into('f', args, 0x200, 1.0)                # q_scale
    struct.pack_into('f', args, 0x204, 1.0)                # k_scale
    struct.pack_into('f', args, 0x208, 1.0)                # v_scale
    
    args_gpu = torch.from_numpy(np.frombuffer(args, dtype=np.uint8)).cuda()
    
    kernarg_ptr = ctypes.c_void_p(args_gpu.data_ptr())
    kernarg_size = ctypes.c_size_t(528)
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
        HIP_LAUNCH_PARAM_END
    )
    
    # Launch kernel (64 threads = 1 wave for now)
    err = hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,    # grid
        64, 1, 1,   # block (1 wave)
        32768,      # shared mem
        None,
        None,
        extra
    )
    if err != 0:
        print(f"hipModuleLaunchKernel failed: {err}")
        return False
    
    hip.hipDeviceSynchronize()
    
    # Get results
    O_kernel = output.cpu()
    O_ref_cpu = O_ref.cpu()
    
    # Check for NaN/Inf
    if torch.isnan(O_kernel).any() or torch.isinf(O_kernel).any():
        print("ERROR: Output contains NaN or Inf!")
        nan_count = torch.isnan(O_kernel).sum().item()
        inf_count = torch.isinf(O_kernel).sum().item()
        print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
        return False
    
    # Check how much output is filled
    nonzero_count = (O_kernel != 0).sum().item()
    total_elements = O_kernel.numel()
    print(f"\nOutput coverage: {nonzero_count}/{total_elements} ({100*nonzero_count/total_elements:.1f}%)")
    
    # Only compare filled portion (first 32 rows for single wave)
    O_kernel_sub = O_kernel[:32, :32]  # Single D-tile
    O_ref_sub = O_ref_cpu[:32, :32]
    
    # Compute error metrics
    abs_diff = (O_kernel_sub - O_ref_sub).abs()
    rel_diff = abs_diff / (O_ref_sub.abs() + 1e-6)
    
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    
    # Correlation
    O_k_flat = O_kernel_sub.flatten()
    O_r_flat = O_ref_sub.flatten()
    if O_k_flat.std() > 0 and O_r_flat.std() > 0:
        correlation = torch.corrcoef(torch.stack([O_k_flat, O_r_flat]))[0, 1].item()
    else:
        correlation = float('nan')
    
    print(f"\nResults (first 32×32 tile):")
    print(f"  Max abs error: {max_abs_err:.6f}")
    print(f"  Mean abs error: {mean_abs_err:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    
    print(f"\nSample outputs (row 0, elements 0:4):")
    print(f"  Reference: {O_ref_sub[0, :4].tolist()}")
    print(f"  Kernel:    {O_kernel_sub[0, :4].tolist()}")
    
    # For v_by_d test
    if test_name == "v_by_d":
        print(f"\nD-dimension check (row 0):")
        expected_d = [d/head_dim for d in range(4)]
        print(f"  Expected ~d/D: {expected_d}")
        print(f"  Kernel:        {O_kernel_sub[0, :4].tolist()}")
    
    # Check pass criteria
    if np.isnan(correlation):
        passed = max_abs_err < 0.1
    else:
        passed = correlation > 0.9 and max_abs_err < 5.0
    
    print(f"\nTest {'PASSED' if passed else 'FAILED'}")
    return passed

def main():
    print("="*60)
    print("FP8 V×P Flash Attention Kernel Tests")
    print("="*60)
    
    all_passed = True
    
    # Test with small dimensions (64 Q rows, 32 K rows)
    seqlen_q, seqlen_k = 64, 32
    
    all_passed &= test_fp8_vxp_kernel(seqlen_q, seqlen_k, test_name="uniform_v")
    all_passed &= test_fp8_vxp_kernel(seqlen_q, seqlen_k, test_name="v_by_k")
    all_passed &= test_fp8_vxp_kernel(seqlen_q, seqlen_k, test_name="v_by_d")
    all_passed &= test_fp8_vxp_kernel(seqlen_q, seqlen_k, test_name="random")
    
    print("\n" + "="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

if __name__ == "__main__":
    main()

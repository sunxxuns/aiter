#!/usr/bin/env python3
"""
Test BF16 flash attention kernel with non-uniform inputs.
Verifies that the V×P computation produces correct results.
"""

import torch
import numpy as np
import ctypes
import struct
import os

torch.manual_seed(42)
np.random.seed(42)

def load_bf16_kernel():
    """Load the BF16 flash attention kernel."""
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    
    co_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16.co"
    module = ctypes.c_void_p()
    err = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if err != 0:
        raise RuntimeError(f"hipModuleLoad failed: {err}")
    
    func = ctypes.c_void_p()
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter19fmha_fwd_hd128_bf16E")
    if err != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {err}")
    
    return hip, func

def compute_reference(Q, K, V, softmax_scale):
    """Compute reference flash attention output."""
    # Q: [seqlen_q, head_dim], K: [seqlen_k, head_dim], V: [seqlen_k, head_dim]
    QK = Q @ K.T  # [seqlen_q, seqlen_k]
    QK = QK * softmax_scale
    P = torch.softmax(QK, dim=-1)  # [seqlen_q, seqlen_k]
    O = P @ V  # [seqlen_q, head_dim]
    return O, P

def test_bf16_kernel(seqlen_q=64, seqlen_k=64, head_dim=128, test_name="random"):
    """Test BF16 kernel with specific input pattern."""
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, head_dim={head_dim}")
    print('='*60)
    
    # Initialize CUDA
    _ = torch.zeros(1, device='cuda')
    
    # Load kernel
    hip, func = load_bf16_kernel()
    
    HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
    HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
    HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)
    
    softmax_scale = 1.0 / np.sqrt(head_dim)
    
    # Generate test data based on pattern
    if test_name == "uniform_v":
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        V = torch.ones(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
    elif test_name == "v_by_k":
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        # V[k, :] = k (V varies by K position)
        V = torch.arange(seqlen_k, dtype=torch.float32, device='cuda').unsqueeze(1).expand(seqlen_k, head_dim)
    elif test_name == "v_by_d":
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        # V[:, d] = d (V varies by D position)
        V = torch.arange(head_dim, dtype=torch.float32, device='cuda').unsqueeze(0).expand(seqlen_k, head_dim)
    else:  # random
        Q = torch.randn(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
        K = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
        V = torch.randn(seqlen_k, head_dim, dtype=torch.float32, device='cuda')
    
    # Compute reference
    O_ref, P = compute_reference(Q, K, V, softmax_scale)
    
    # Convert to BF16 for kernel input
    Q_bf16 = Q.to(torch.bfloat16)
    K_bf16 = K.to(torch.bfloat16)
    V_bf16 = V.to(torch.bfloat16)
    
    # Allocate output
    output = torch.zeros(seqlen_q, head_dim, dtype=torch.float32, device='cuda')
    LSE = torch.zeros(seqlen_q, dtype=torch.float32, device='cuda')  # log-sum-exp
    
    # Pack kernel arguments (based on BF16 kernel layout)
    args = bytearray(512)
    struct.pack_into('Q', args, 0x00, output.data_ptr())  # ptr_R (output)
    struct.pack_into('Q', args, 0x10, Q_bf16.data_ptr())  # ptr_Q
    struct.pack_into('Q', args, 0x20, K_bf16.data_ptr())  # ptr_K
    struct.pack_into('Q', args, 0x30, V_bf16.data_ptr())  # ptr_V
    struct.pack_into('Q', args, 0x40, LSE.data_ptr())     # ptr_LSE
    struct.pack_into('f', args, 0x50, softmax_scale)       # softmax_scale
    struct.pack_into('I', args, 0x60, seqlen_q)            # seqlen_q
    struct.pack_into('I', args, 0x70, seqlen_k)            # seqlen_k
    struct.pack_into('I', args, 0x80, head_dim)            # head_dim
    # Strides (assuming contiguous row-major)
    struct.pack_into('I', args, 0x90, head_dim)            # Q row stride
    struct.pack_into('I', args, 0xa0, head_dim)            # K row stride  
    struct.pack_into('I', args, 0xb0, seqlen_k * head_dim) # batch stride Q
    struct.pack_into('I', args, 0xc0, seqlen_k * head_dim) # batch stride K
    struct.pack_into('I', args, 0xd0, seqlen_k * head_dim) # batch stride V
    struct.pack_into('I', args, 0xe0, seqlen_q * head_dim) # batch stride O
    struct.pack_into('I', args, 0xf0, seqlen_q)            # LSE stride
    
    args_gpu = torch.from_numpy(np.frombuffer(args, dtype=np.uint8)).cuda()
    
    # Calculate grid/block dimensions
    # BF16 kernel uses 256 threads (4 waves), processes BLOCK_M=64 Q rows
    num_blocks_q = (seqlen_q + 63) // 64  # ceil(seqlen_q / 64)
    
    kernarg_ptr = ctypes.c_void_p(args_gpu.data_ptr())
    kernarg_size = ctypes.c_size_t(512)
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
        HIP_LAUNCH_PARAM_END
    )
    
    # Launch kernel
    err = hip.hipModuleLaunchKernel(
        func,
        num_blocks_q, 1, 1,  # grid
        256, 1, 1,           # block (4 waves)
        32768,               # shared mem
        None,                # stream
        None,                # kernelParams (use extra instead)
        extra
    )
    if err != 0:
        print(f"hipModuleLaunchKernel failed: {err}")
        return False
    
    hip.hipDeviceSynchronize()
    
    # Compare outputs
    O_kernel = output.cpu()
    O_ref_cpu = O_ref.cpu()
    
    # Check for NaN/Inf
    if torch.isnan(O_kernel).any() or torch.isinf(O_kernel).any():
        print("ERROR: Output contains NaN or Inf!")
        print(f"NaN count: {torch.isnan(O_kernel).sum().item()}")
        print(f"Inf count: {torch.isinf(O_kernel).sum().item()}")
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
    print(f"  Max rel error: {max_rel_err:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # Show sample values
    print(f"\nSample outputs (first row, first 4 elements):")
    print(f"  Reference: {O_ref_cpu[0, :4].tolist()}")
    print(f"  Kernel:    {O_kernel[0, :4].tolist()}")
    
    # Check if test passes (BF16 has limited precision)
    # Allow larger tolerance for BF16 (relative error up to ~1% is acceptable)
    passed = correlation > 0.99 and max_abs_err < 1.0
    print(f"\nTest {'PASSED' if passed else 'FAILED'}")
    
    return passed

def main():
    """Run all tests."""
    print("="*60)
    print("BF16 Flash Attention Kernel Numeric Tests")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Uniform V (should give mean of V = 1.0)
    all_passed &= test_bf16_kernel(test_name="uniform_v")
    
    # Test 2: V varies by K (V[k,:] = k)
    all_passed &= test_bf16_kernel(test_name="v_by_k")
    
    # Test 3: V varies by D (V[:,d] = d)
    all_passed &= test_bf16_kernel(test_name="v_by_d")
    
    # Test 4: Random inputs
    all_passed &= test_bf16_kernel(test_name="random")
    
    print("\n" + "="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

if __name__ == "__main__":
    main()

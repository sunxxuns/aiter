#!/usr/bin/env python3
"""
Test FP8 Flash Attention Assembly Kernel
Verifies: O = softmax(Q @ K^T) @ V

Usage: python test_fp8_attention.py
"""

import torch
import numpy as np
import ctypes
import struct
import subprocess
import sys
import os

# Configuration
KERNEL_FILE = "integrate_step3_hd32.s"
KERNEL_CO = "integrate_step3_hd32.co"
KERNEL_SYMBOL = "_ZN5aiter20integrate_step3_hd32E"
SEQ_LEN = 32
HEAD_DIM = 32

def build_kernel():
    """Build the assembly kernel"""
    print("Building kernel...")
    result = subprocess.run(
        ['/opt/rocm/llvm/bin/clang++', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
         '-mcpu=gfx950', '-o', KERNEL_CO, KERNEL_FILE],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Build FAILED:\n{result.stderr}")
        return False
    print("Build successful!")
    return True

class FP8AttentionKernel:
    def __init__(self):
        self.hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
        # Initialize CUDA/HIP context
        _ = torch.zeros(1, device='cuda')

        # Load module
        self.module = ctypes.c_void_p()
        ret = self.hip.hipModuleLoad(ctypes.byref(self.module), KERNEL_CO.encode())
        if ret != 0:
            raise RuntimeError(f"hipModuleLoad failed: {ret}")

        # Get function
        self.func = ctypes.c_void_p()
        ret = self.hip.hipModuleGetFunction(ctypes.byref(self.func), self.module, KERNEL_SYMBOL.encode())
        if ret != 0:
            raise RuntimeError(f"hipModuleGetFunction failed: {ret}")

    def __call__(self, Q_fp8, K_fp8, V_fp8):
        """Run kernel: O = softmax(Q @ K^T) @ V"""
        O_out = torch.zeros(SEQ_LEN, HEAD_DIM, dtype=torch.float32, device='cuda')

        # Pack kernel arguments
        args = bytearray(32)
        struct.pack_into('Q', args, 0, O_out.data_ptr())
        struct.pack_into('Q', args, 8, Q_fp8.data_ptr())
        struct.pack_into('Q', args, 16, K_fp8.data_ptr())
        struct.pack_into('Q', args, 24, V_fp8.data_ptr())

        # Launch parameters
        args_gpu = torch.from_numpy(np.frombuffer(args, dtype=np.uint8)).cuda()
        HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
        HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
        HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

        kernarg_size = ctypes.c_size_t(32)
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.c_void_p(args_gpu.data_ptr()),
            HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
            HIP_LAUNCH_PARAM_END
        )

        # Launch kernel
        self.hip.hipModuleLaunchKernel(
            self.func,
            1, 1, 1,      # grid
            64, 1, 1,     # block
            6144,         # shared memory
            None, None, extra
        )
        self.hip.hipDeviceSynchronize()

        return O_out

def to_fp8(x):
    """Convert float tensor to FP8 (e4m3fn format)"""
    return x.to(torch.float8_e4m3fn).view(torch.uint8)

def from_fp8(x):
    """Convert FP8 bytes back to float"""
    return x.view(torch.float8_e4m3fn).float()

def pytorch_attention(Q, K, V):
    """Reference implementation using PyTorch"""
    S = Q @ K.T
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O

def run_test(name, Q, K, V, kernel, verbose=True):
    """Run single test case"""
    # Convert to FP8
    Q_fp8 = to_fp8(Q).cuda()
    K_fp8 = to_fp8(K).cuda()
    V_fp8 = to_fp8(V).cuda()

    # Get quantized values for reference
    Q_q = from_fp8(Q_fp8.cpu())
    K_q = from_fp8(K_fp8.cpu())
    V_q = from_fp8(V_fp8.cpu())

    # PyTorch reference
    O_ref = pytorch_attention(Q_q, K_q, V_q)

    # Kernel output
    O_kernel = kernel(Q_fp8, K_fp8, V_fp8).cpu()
    print(f"{kernel}, {O_kernel}")

    # Metrics
    diff = (O_kernel - O_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    has_nan = torch.isnan(O_kernel).any().item()
    has_inf = torch.isinf(O_kernel).any().item()

    # Correlation (handle uniform case)
    if O_ref.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([O_ref.flatten(), O_kernel.flatten()]))[0,1].item()
    else:
        corr = 1.0 if max_err < 1e-5 else 0.0

    passed = max_err < 0.05 and (corr > 0.99 or max_err < 0.001) and not has_nan and not has_inf

    if verbose:
        print(f"\n  [{name}]")
        print(f"  PyTorch O[0,:4]: {O_ref[0,:4].tolist()}")
        print(f"  Kernel  O[0,:4]: {O_kernel[0,:4].tolist()}")
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Correlation: {corr:.6f}")
        if has_nan: print("  WARNING: Contains NaN!")
        if has_inf: print("  WARNING: Contains Inf!")
        print(f"  {'✓ PASSED' if passed else '✗ FAILED'}")

    return passed, max_err, corr

def main():
    print("=" * 60)
    print("FP8 Flash Attention Kernel Test")
    print(f"seq_len={SEQ_LEN}, head_dim={HEAD_DIM}")
    print("=" * 60)

    # Build kernel
    if not build_kernel():
        return 1

    # Load kernel
    print("\nLoading kernel...")
    kernel = FP8AttentionKernel()
    print("Kernel loaded!")

    print("\n" + "-" * 60)
    print("Running tests...")
    print("-" * 60)

    results = []

    # Test 1: Uniform inputs
    Q = torch.ones(SEQ_LEN, HEAD_DIM)
    K = torch.ones(SEQ_LEN, HEAD_DIM)
    V = torch.ones(SEQ_LEN, HEAD_DIM)
    results.append(run_test("Uniform Q=K=V=1", Q, K, V, kernel))

    # Test 2: Random small
    torch.manual_seed(42)
    Q = torch.randn(SEQ_LEN, HEAD_DIM) * 0.3
    K = torch.randn(SEQ_LEN, HEAD_DIM) * 0.3
    V = torch.randn(SEQ_LEN, HEAD_DIM) * 0.3
    results.append(run_test("Random (seed=42, scale=0.3)", Q, K, V, kernel))

    # Test 3: Random medium
    torch.manual_seed(123)
    Q = torch.randn(SEQ_LEN, HEAD_DIM) * 0.5
    K = torch.randn(SEQ_LEN, HEAD_DIM) * 0.5
    V = torch.randn(SEQ_LEN, HEAD_DIM) * 0.5
    results.append(run_test("Random (seed=123, scale=0.5)", Q, K, V, kernel))

    # Test 4: Positive V
    torch.manual_seed(456)
    Q = torch.randn(SEQ_LEN, HEAD_DIM) * 0.4
    K = torch.randn(SEQ_LEN, HEAD_DIM) * 0.4
    V = torch.rand(SEQ_LEN, HEAD_DIM) * 0.5 + 0.25
    results.append(run_test("Random Q,K with positive V", Q, K, V, kernel))

    # Test 5: Identity V
    Q = torch.ones(SEQ_LEN, HEAD_DIM) * 0.5
    K = torch.ones(SEQ_LEN, HEAD_DIM) * 0.5
    V = torch.eye(SEQ_LEN)
    results.append(run_test("V=identity", Q, K, V, kernel))

    # Test 6: Additional random
    torch.manual_seed(789)
    Q = torch.randn(SEQ_LEN, HEAD_DIM) * 0.4
    K = torch.randn(SEQ_LEN, HEAD_DIM) * 0.4
    V = torch.randn(SEQ_LEN, HEAD_DIM) * 0.4
    results.append(run_test("Random (seed=789, scale=0.4)", Q, K, V, kernel))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r[0])
    total = len(results)

    print(f"\nPassed: {passed}/{total}")
    print(f"Max error across all tests: {max(r[1] for r in results):.6f}")
    print(f"Min correlation: {min(r[2] for r in results):.6f}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

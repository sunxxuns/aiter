#!/usr/bin/env python3
"""
Test minimal PV MFMA kernel.
With V=1 and P=1, output should be K (the number of K values reduced).
"""

import torch
import numpy as np
import ctypes
import struct
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_kernel():
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    
    co_path = os.path.join(SCRIPT_DIR, "test_pv_mfma_minimal.co")
    module = ctypes.c_void_p()
    err = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if err != 0:
        raise RuntimeError(f"hipModuleLoad failed: {err}")
    
    func = ctypes.c_void_p()
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter18test_pv_mfma_minE")
    if err != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {err}")
    
    return hip, func

def main():
    print("="*60)
    print("Minimal PV MFMA Test")
    print("="*60)
    
    # Initialize CUDA
    _ = torch.zeros(1, device='cuda')
    
    hip, func = load_kernel()
    
    HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
    HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
    HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)
    
    # Output: 32 Q rows × 128 D cols (but only 32 D cols filled by 1 wave)
    output = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    dummy_v = torch.zeros(1, device='cuda')
    dummy_p = torch.zeros(1, device='cuda')
    mode = 0  # V=1, P=1 -> output should be 16
    
    # Pack kernel arguments
    args = bytearray(32)
    struct.pack_into('Q', args, 0, output.data_ptr())
    struct.pack_into('Q', args, 8, dummy_v.data_ptr())
    struct.pack_into('Q', args, 16, dummy_p.data_ptr())
    struct.pack_into('I', args, 24, mode)
    
    args_gpu = torch.from_numpy(np.frombuffer(args, dtype=np.uint8)).cuda()
    
    kernarg_ptr = ctypes.c_void_p(args_gpu.data_ptr())
    kernarg_size = ctypes.c_size_t(32)
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(kernarg_size),
        HIP_LAUNCH_PARAM_END
    )
    
    # Launch kernel (1 wave = 64 threads)
    err = hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,    # grid
        64, 1, 1,   # block
        0,          # shared mem
        None,
        None,
        extra
    )
    if err != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {err}")
    
    hip.hipDeviceSynchronize()
    
    # Check results
    O = output.cpu()
    
    print(f"\nWith V=1, P=1, K=16 (one MFMA tile):")
    print(f"Expected output: 16.0 (K reduction)")
    print(f"Actual output[0, 0]: {O[0, 0].item()}")
    print(f"Actual output[0, 1]: {O[0, 1].item()}")
    print(f"Actual output[1, 0]: {O[1, 0].item()}")
    
    print(f"\nFirst 4x4 tile:")
    print(O[:4, :4])
    
    # Check statistics
    filled = O[:32, :32]  # 32x32 tile
    print(f"\nStatistics (32×32 tile):")
    print(f"  Min: {filled.min().item()}")
    print(f"  Max: {filled.max().item()}")
    print(f"  Mean: {filled.mean().item()}")
    print(f"  Unique values: {filled.unique()}")
    
    # Expected: all 16.0
    expected = 16.0
    if torch.allclose(filled, torch.full_like(filled, expected), atol=1.0):
        print(f"\n✓ Output matches expected value ~{expected}")
    else:
        print(f"\n✗ Output does NOT match expected value {expected}")
        # Check if there's a pattern
        print(f"  Row 0: {O[0, :8].tolist()}")
        print(f"  Row 1: {O[1, :8].tolist()}")
        print(f"  Col 0: {O[:8, 0].tolist()}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode the MFMA output layout by using identity-like matrices."""

import torch
import subprocess
import ctypes
from pathlib import Path

def build_kernel():
    src_dir = Path(__file__).parent
    src_file = src_dir / "test_mfma_raw.s"
    obj_file = src_dir / "test_mfma_raw.o"
    co_file = src_dir / "test_mfma_raw.co"
    
    cmd1 = f"clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c {src_file} -o {obj_file}"
    subprocess.run(cmd1, shell=True, capture_output=True)
    cmd2 = f"clang++ -target amdgcn-amd-amdhsa -mcpu=gfx950 {obj_file} -o {co_file}"
    subprocess.run(cmd2, shell=True, capture_output=True)
    return co_file

def run_mfma(hip, func, Q, K):
    O = torch.zeros(64, 16, dtype=torch.float32, device='cuda')
    
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(ctypes.c_void_p(O.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_void_p(Q.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_void_p(K.data_ptr())), ctypes.c_void_p),
    )
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 2048, None, args, None)
    hip.hipDeviceSynchronize()
    return O

def main():
    print("=" * 70)
    print("Decode MFMA output layout")
    print("=" * 70)
    
    hip = ctypes.CDLL("libamdhip64.so")
    co_file = build_kernel()
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), str(co_file).encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13test_mfma_rawE")
    
    # Build mapping: for each S[r,c], find which thread/vreg contains it
    # Strategy: set Q[r,:] = all 1s, K[c,:] = all 1s, all else 0
    # Then S[r,c] = 16 (sum of 16 ones)
    
    mapping = {}  # (r, c) -> (tid, vreg)
    
    for target_row in range(32):
        for target_col in range(32):
            Q = torch.zeros(32, 32, dtype=torch.uint8, device='cuda')
            K = torch.zeros(32, 32, dtype=torch.uint8, device='cuda')
            
            # Set Q[target_row, 0:16] = 1.0 (FP8)
            # Set K[target_col, 0:16] = 1.0 (FP8)
            Q[target_row, 0:16] = 0x3C  # 1.5 in FP8
            K[target_col, 0:16] = 0x3C  # 1.5 in FP8
            
            # Expected: S[target_row, target_col] = 16 * 1.5 * 1.5 = 36
            O = run_mfma(hip, func, Q, K)
            
            # Find where 36 appears
            target_val = 16 * 1.5 * 1.5  # 36
            matches = (O.abs() - target_val).abs() < 1.0
            if matches.any():
                idx = matches.nonzero()[0]
                tid, vreg = idx[0].item(), idx[1].item()
                mapping[(target_row, target_col)] = (tid, 32 + vreg)
    
    # Analyze the mapping
    print("\nMFMA output layout: S[row, col] -> Thread tid, v{vreg}")
    print("-" * 70)
    
    # Print mapping for first few rows/cols
    print("\n   col: ", end="")
    for c in range(8):
        print(f"{c:8d}", end="")
    print()
    
    for r in range(16):
        print(f"row {r:2d}: ", end="")
        for c in range(8):
            if (r, c) in mapping:
                tid, vreg = mapping[(r, c)]
                print(f"({tid:2d},v{vreg})", end=" ")
            else:
                print("  ???   ", end=" ")
        print()
    
    # Derive the formula
    print("\n\nDeriving store pattern formula...")
    print("-" * 70)
    
    # Group by (tid, vreg) to see which S elements each produces
    from collections import defaultdict
    tid_vreg_to_elements = defaultdict(list)
    for (r, c), (tid, vreg) in mapping.items():
        tid_vreg_to_elements[(tid, vreg)].append((r, c))
    
    # Show what each vreg produces for thread 0
    print("\nThread 0's output elements:")
    for vreg in range(32, 48):
        elements = tid_vreg_to_elements.get((0, vreg), [])
        if elements:
            print(f"  v{vreg}: S{elements}")
    
    print("\nThread 32's output elements:")
    for vreg in range(32, 48):
        elements = tid_vreg_to_elements.get((32, vreg), [])
        if elements:
            print(f"  v{vreg}: S{elements}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

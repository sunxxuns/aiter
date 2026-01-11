#!/usr/bin/env python3
"""Test QK with fixed store pattern."""

import torch
import subprocess
import ctypes
from pathlib import Path

def build_kernel():
    src_dir = Path(__file__).parent
    src_file = src_dir / "test_qk_fixed.s"
    obj_file = src_dir / "test_qk_fixed.o"
    co_file = src_dir / "test_qk_fixed.co"
    
    cmd1 = f"clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c {src_file} -o {obj_file}"
    result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Assembly failed:\n{result.stderr}")
        return None
    
    cmd2 = f"clang++ -target amdgcn-amd-amdhsa -mcpu=gfx950 {obj_file} -o {co_file}"
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Linking failed:\n{result.stderr}")
        return None
    
    return co_file

def run_test():
    print("=" * 70)
    print("Test: QK with FIXED store pattern (buffer_load→LDS)")
    print("=" * 70)
    
    co_file = build_kernel()
    if co_file is None:
        return False
    
    hip = ctypes.CDLL("libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), str(co_file).encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13test_qk_fixedE")
    
    # Test with random FP8 data
    torch.manual_seed(42)
    Q_f32 = torch.randn(32, 32, device='cuda') * 0.5
    K_f32 = torch.randn(32, 32, device='cuda') * 0.5
    
    Q_fp8 = Q_f32.to(torch.float8_e4m3fn)
    K_fp8 = K_f32.to(torch.float8_e4m3fn)
    
    S_out = torch.zeros(32, 32, dtype=torch.float32, device='cuda')
    
    # Reference
    Q_ref = Q_fp8.to(torch.float32)
    K_ref = K_fp8.to(torch.float32)
    S_ref = Q_ref @ K_ref.T
    
    # Launch
    ptr_S = ctypes.c_void_p(S_out.data_ptr())
    ptr_Q = ctypes.c_void_p(Q_fp8.data_ptr())
    ptr_K = ctypes.c_void_p(K_fp8.data_ptr())
    
    args = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.pointer(ptr_S), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ptr_Q), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ptr_K), ctypes.c_void_p),
    )
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 2048, None, args, None)
    hip.hipDeviceSynchronize()
    
    # Compare
    print(f"\nS_ref[0,:8]: {S_ref[0,:8].tolist()}")
    print(f"S_out[0,:8]: {S_out[0,:8].tolist()}")
    print(f"\nS_ref[1,:8]: {S_ref[1,:8].tolist()}")
    print(f"S_out[1,:8]: {S_out[1,:8].tolist()}")
    
    has_nan = torch.isnan(S_out).any().item()
    has_inf = torch.isinf(S_out).any().item()
    
    diff = (S_out - S_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    print(f"\nNaN: {has_nan}, Inf: {has_inf}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    passed = max_err < 1e-3 and not has_nan and not has_inf  # FP8 precision tolerance
    print(f"\nPASSED: {passed}")
    
    print("=" * 70)
    return passed

if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""Test for integrate_step4_hd128.s - FP8 Flash Attention with head_dim=128"""

import torch
import subprocess
import os
import ctypes
import numpy as np

# Build the kernel
def build_kernel():
    src = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.s"
    obj = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.o"
    co = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.co"
    
    # Assemble
    cmd = f"/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c {src} -o {obj}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Assembly failed:")
        print(result.stderr)
        return False
    
    # Link
    cmd = f"/opt/rocm/llvm/bin/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx950 {obj} -o {co}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Linking failed:")
        print(result.stderr)
        return False
    
    print(f"Built {co}")
    return True

def load_kernel():
    co_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/integrate_step4_hd128.co"
    
    hip = ctypes.CDLL("libamdhip64.so")
    
    # Load module
    module = ctypes.c_void_p()
    status = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if status != 0:
        raise RuntimeError(f"hipModuleLoad failed with status {status}")
    
    # Get function
    function = ctypes.c_void_p()
    status = hip.hipModuleGetFunction(
        ctypes.byref(function), 
        module, 
        b"_ZN5aiter21integrate_step4_hd128E"
    )
    if status != 0:
        raise RuntimeError(f"hipModuleGetFunction failed with status {status}")
    
    return hip, module, function

def launch_kernel(hip, function, O, Q, K, V):
    """Launch kernel with 64 threads (one wavefront)"""
    
    # Use kernelParams style - array of pointers to arguments
    args_array = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(O.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(Q.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(K.data_ptr())), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(ctypes.c_uint64(V.data_ptr())), ctypes.c_void_p),
    )
    
    # Launch: 1 block of 64 threads
    status = hip.hipModuleLaunchKernel(
        function,
        1, 1, 1,      # grid
        64, 1, 1,     # block (one wavefront)
        6144,         # shared mem
        None,         # stream
        args_array,   # kernel params
        None          # extra
    )
    if status != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed with status {status}")
    
    hip.hipDeviceSynchronize()

def reference_attention(Q, K, V):
    """PyTorch reference implementation"""
    # Q, K, V are [32, 128] FP8 stored as uint8, need to convert to float
    # Use PyTorch's native FP8 support
    Q_f32 = Q.view(torch.float8_e4m3fn).to(torch.float32)
    K_f32 = K.view(torch.float8_e4m3fn).to(torch.float32)
    V_f32 = V.view(torch.float8_e4m3fn).to(torch.float32)
    
    # S = Q @ K^T
    S = torch.matmul(Q_f32, K_f32.T)
    
    # Softmax
    P = torch.softmax(S, dim=-1)
    
    # O = P @ V
    O = torch.matmul(P, V_f32)
    
    return O

def run_test(name, Q, K, V):
    """Run a single test case"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    
    # Create output buffer
    O = torch.zeros(32, 128, dtype=torch.float32, device='cuda')
    
    # Load and run kernel
    hip, module, function = load_kernel()
    launch_kernel(hip, function, O, Q, K, V)
    
    # Get reference
    O_ref = reference_attention(Q, K, V)
    
    # Compare
    O_cpu = O.cpu().numpy()
    O_ref_cpu = O_ref.cpu().numpy()
    
    max_err = np.abs(O_cpu - O_ref_cpu).max()
    mean_err = np.abs(O_cpu - O_ref_cpu).mean()
    
    # Correlation
    corr = np.corrcoef(O_cpu.flatten(), O_ref_cpu.flatten())[0, 1]
    
    print(f"Max error:  {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    print(f"Correlation: {corr:.6f}")
    
    # Show sample values
    print(f"\nSample O[0, :8]:")
    print(f"  Kernel: {O_cpu[0, :8]}")
    print(f"  Ref:    {O_ref_cpu[0, :8]}")
    
    print(f"\nSample O[16, 64:72]:")
    print(f"  Kernel: {O_cpu[16, 64:72]}")
    print(f"  Ref:    {O_ref_cpu[16, 64:72]}")
    
    # Handle NaN correlation (when all values are identical)
    if np.isnan(corr):
        corr = 1.0 if max_err < 0.001 else 0.0
    passed = max_err < 0.1 and corr > 0.99
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    
    hip.hipModuleUnload(module)
    return passed

def main():
    print("Building kernel...")
    if not build_kernel():
        return
    
    print("\nRunning tests...")
    
    results = []
    
    # Test 1: Uniform values
    Q = torch.full((32, 128), 0.5, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = torch.full((32, 128), 0.5, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = torch.full((32, 128), 0.5, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    results.append(run_test("Uniform 0.5", Q, K, V))
    
    # Test 2: Random small values
    torch.manual_seed(42)
    Q = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    results.append(run_test("Random small", Q, K, V))
    
    # Test 3: Random larger values
    torch.manual_seed(123)
    Q = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V = (torch.randn(32, 128) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    results.append(run_test("Random larger", Q, K, V))
    
    # Test 4: Identity-like V (diagonal pattern)
    Q = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    K = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    V_f = torch.zeros(32, 128)
    for i in range(32):
        V_f[i, i*4:(i+1)*4] = 1.0
    V = V_f.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    results.append(run_test("Structured V", Q, K, V))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test: Is our kernel doing ROW-WISE softmax (correct) or BLOCK-WISE (wrong)?

Standard attention requires:
  P[i,:] = softmax(S[i,:])  # Each query row normalized independently

This test checks by using inputs where row-wise vs block-wise give very different results.
"""

import torch
import subprocess
import ctypes
import os
import math

def build_kernel():
    src = "test_full_hd128.s"
    obj = "test_full_hd128.o"
    out = "test_full_hd128.co"
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", src, "-o", obj],
        capture_output=True, cwd=cwd
    )
    subprocess.run(
        ["ld.lld", "-shared", "-o", out, obj],
        capture_output=True, cwd=cwd
    )
    return os.path.join(cwd, out)


def run_kernel(Q_fp8, K_fp8, V_fp8, hip, func):
    SEQ, HD = Q_fp8.shape
    O = torch.zeros(SEQ, HD, dtype=torch.float32, device='cuda')
    
    args_list = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q_fp8.data_ptr()),
        ctypes.c_void_p(K_fp8.data_ptr()),
        ctypes.c_void_p(V_fp8.data_ptr()),
    ]
    args_array = (ctypes.c_void_p * len(args_list))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args_list]
    )
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 65536, None, args_array, None)
    hip.hipDeviceSynchronize()
    return O


def compute_rowwise_reference(Q_fp8, K_fp8, V_fp8):
    """Correct attention: row-wise softmax"""
    Q = Q_fp8.float()
    K = K_fp8.float()
    V = V_fp8.float()
    HD = Q.shape[1]
    
    S = Q @ K.T
    scale = 1.0 / math.sqrt(HD)
    
    # Row-wise softmax: each row has its own max and sum
    P = torch.softmax(S * scale, dim=1)  # dim=1 = over keys
    
    # Quantize P to FP8 like kernel does
    P_fp8 = P.to(torch.float8_e4m3fn).float()
    
    # Normalize again after FP8 quantization (row-wise)
    P_fp8_norm = P_fp8 / P_fp8.sum(dim=1, keepdim=True)
    
    O = P_fp8_norm @ V
    return O


def compute_blockwise_reference(Q_fp8, K_fp8, V_fp8):
    """Wrong attention: block-wise (global) softmax"""
    Q = Q_fp8.float()
    K = K_fp8.float()
    V = V_fp8.float()
    HD = Q.shape[1]
    
    S = Q @ K.T
    scale = 1.0 / math.sqrt(HD)
    
    # Block-wise softmax: global max/sum over entire S matrix
    S_scaled = S * scale
    global_max = S_scaled.max()
    P_unnorm = torch.exp(S_scaled - global_max)
    global_sum = P_unnorm.sum()
    
    # Quantize P to FP8
    P_fp8 = P_unnorm.to(torch.float8_e4m3fn).float()
    P_fp8_sum = P_fp8.sum()
    
    O = (P_fp8 @ V) / P_fp8_sum
    return O


def main():
    print("=" * 70)
    print("ROW-WISE vs BLOCK-WISE SOFTMAX TEST")
    print("=" * 70)
    
    co_path = build_kernel()
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter14test_full_hd128E")
    
    SEQ, HD = 32, 128
    
    # Test case: Different rows have very different attention patterns
    # Row 0: attends strongly to key 0
    # Row 1: attends strongly to key 1
    # etc.
    print("\nTest: Diagonal-dominant attention pattern")
    print("-" * 50)
    
    # Create Q and K such that Q[i] is most similar to K[i]
    Q = torch.zeros(SEQ, HD, device='cuda')
    K = torch.zeros(SEQ, HD, device='cuda')
    V = torch.zeros(SEQ, HD, device='cuda')
    
    # Each Q[i] has a unique signature that matches K[i]
    for i in range(SEQ):
        Q[i, :] = 0.1
        Q[i, i % HD] = 2.0  # Strong signal in one dimension
        K[i, :] = 0.1  
        K[i, i % HD] = 2.0  # Matching signal
        V[i, :] = float(i) / SEQ  # Different value for each key
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    # Compute references
    O_rowwise = compute_rowwise_reference(Q_fp8, K_fp8, V_fp8)
    O_blockwise = compute_blockwise_reference(Q_fp8, K_fp8, V_fp8)
    
    # Run kernel
    O_kernel = run_kernel(Q_fp8, K_fp8, V_fp8, hip, func)
    
    # Compare
    err_vs_rowwise = (O_kernel - O_rowwise).abs().max().item()
    err_vs_blockwise = (O_kernel - O_blockwise).abs().max().item()
    
    print(f"Kernel vs ROW-WISE ref:   max_err = {err_vs_rowwise:.6f}")
    print(f"Kernel vs BLOCK-WISE ref: max_err = {err_vs_blockwise:.6f}")
    
    print(f"\nO_kernel[0,:4]:    {O_kernel[0,:4].tolist()}")
    print(f"O_rowwise[0,:4]:   {O_rowwise[0,:4].tolist()}")
    print(f"O_blockwise[0,:4]: {O_blockwise[0,:4].tolist()}")
    
    print(f"\nO_kernel[15,:4]:    {O_kernel[15,:4].tolist()}")
    print(f"O_rowwise[15,:4]:   {O_rowwise[15,:4].tolist()}")
    print(f"O_blockwise[15,:4]: {O_blockwise[15,:4].tolist()}")
    
    if err_vs_rowwise < err_vs_blockwise:
        print("\n✅ Kernel is closer to ROW-WISE (correct)")
        is_rowwise = True
    else:
        print("\n❌ Kernel is closer to BLOCK-WISE (WRONG for attention)")
        is_rowwise = False
    
    # Test 2: More extreme case
    print("\n" + "=" * 70)
    print("Test: Extreme case - one row hot, others cold")
    print("-" * 50)
    
    Q = torch.randn(SEQ, HD, device='cuda') * 0.1
    K = torch.randn(SEQ, HD, device='cuda') * 0.1
    V = torch.arange(SEQ, device='cuda', dtype=torch.float32).unsqueeze(1).expand(SEQ, HD) / SEQ
    
    # Make row 0 of Q very similar to row 0 of K (high attention score)
    Q[0, :] = 1.0
    K[0, :] = 1.0
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    O_rowwise = compute_rowwise_reference(Q_fp8, K_fp8, V_fp8)
    O_blockwise = compute_blockwise_reference(Q_fp8, K_fp8, V_fp8)
    O_kernel = run_kernel(Q_fp8, K_fp8, V_fp8, hip, func)
    
    err_vs_rowwise = (O_kernel - O_rowwise).abs().max().item()
    err_vs_blockwise = (O_kernel - O_blockwise).abs().max().item()
    
    print(f"Kernel vs ROW-WISE ref:   max_err = {err_vs_rowwise:.6f}")
    print(f"Kernel vs BLOCK-WISE ref: max_err = {err_vs_blockwise:.6f}")
    
    print(f"\nRow 0 (should attend to key 0, get V[0]≈0):")
    print(f"  O_kernel[0,0]:    {O_kernel[0,0].item():.6f}")
    print(f"  O_rowwise[0,0]:   {O_rowwise[0,0].item():.6f}")
    print(f"  O_blockwise[0,0]: {O_blockwise[0,0].item():.6f}")
    
    print(f"\nRow 15 (should have uniform attention, get avg V≈0.5):")
    print(f"  O_kernel[15,0]:    {O_kernel[15,0].item():.6f}")
    print(f"  O_rowwise[15,0]:   {O_rowwise[15,0].item():.6f}")  
    print(f"  O_blockwise[15,0]: {O_blockwise[15,0].item():.6f}")
    
    if err_vs_rowwise < err_vs_blockwise:
        print("\n✅ Kernel is closer to ROW-WISE (correct)")
    else:
        print("\n❌ Kernel is closer to BLOCK-WISE (WRONG)")
    
    hip.hipModuleUnload(module)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test ONLY the softmax part - output P matrix to verify row-wise behavior.
Skip PV MFMA entirely.
"""

import torch
import subprocess
import ctypes
import os
import math

def build_and_run():
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    
    # Build kernel
    subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "test_full_hd128.s", "-o", "test_full_hd128.o"],
        capture_output=True, cwd=cwd, check=True
    )
    subprocess.run(
        ["ld.lld", "-shared", "-o", "test_full_hd128.co", "test_full_hd128.o"],
        capture_output=True, cwd=cwd, check=True
    )
    
    co_path = os.path.join(cwd, "test_full_hd128.co")
    
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter14test_full_hd128E")
    
    SEQ, HD = 32, 128
    
    # Simple test: identity-like pattern
    # Q[i] similar to K[i] -> S[i,i] should be largest in row i
    print("=" * 60)
    print("SOFTMAX CHECK: Is it row-wise or block-wise?")
    print("=" * 60)
    
    Q = torch.zeros(SEQ, HD, device='cuda')
    K = torch.zeros(SEQ, HD, device='cuda')
    V = torch.eye(SEQ, HD, device='cuda')  # Identity-like V
    
    # Make Q[i] most similar to K[i]
    for i in range(SEQ):
        Q[i, :] = 0.1
        K[i, :] = 0.1
        Q[i, i % HD] = 2.0
        K[i, i % HD] = 2.0
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
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
    
    # Compute references
    scale = 1.0 / math.sqrt(HD)
    Q_f = Q_fp8.float()
    K_f = K_fp8.float()
    
    # S matrix
    S = Q_f @ K_f.T
    S_scaled = S * scale
    
    print(f"\nS[0,:8] (query 0's scores): {S_scaled[0,:8].tolist()}")
    print(f"S[0,0] should be highest in row 0: {S_scaled[0,0].item():.4f}")
    
    # Row-wise softmax
    P_rowwise = torch.softmax(S_scaled, dim=1)
    print(f"\nRow-wise P[0,:8]: {P_rowwise[0,:8].tolist()}")
    print(f"Row-wise P[0].sum() = {P_rowwise[0].sum().item():.6f}")
    print(f"Row-wise P[0,0] (should be ~0.044): {P_rowwise[0,0].item():.4f}")
    
    # Block-wise softmax  
    global_max = S_scaled.max()
    P_block_unnorm = torch.exp(S_scaled - global_max)
    P_blockwise = P_block_unnorm / P_block_unnorm.sum()
    print(f"\nBlock-wise P[0,:8]: {P_blockwise[0,:8].tolist()}")
    print(f"Block-wise P[0].sum() = {P_blockwise[0].sum().item():.6f}")
    
    # Check kernel output
    print(f"\nKernel O[0,:8]: {O[0,:8].tolist()}")
    print(f"Kernel O[15,:8]: {O[15,:8].tolist()}")
    
    # With identity V and row-wise P:
    # O[i,j] = sum_k P[i,k] * V[k,j] = P[i,j] (since V is identity-like)
    # So O[0,0] should be ~P[0,0] for row-wise
    
    # Row-wise reference O
    V_f = V_fp8.float()
    P_fp8 = P_rowwise.to(torch.float8_e4m3fn).float()
    O_rowwise = P_fp8 @ V_f
    
    print(f"\nRow-wise O[0,:8]: {O_rowwise[0,:8].tolist()}")
    
    # Check which one kernel matches
    err_rowwise = (O - O_rowwise).abs().max().item()
    
    # Block-wise O
    P_block_fp8 = P_blockwise.to(torch.float8_e4m3fn).float()
    O_blockwise = (P_block_fp8 @ V_f) / P_block_fp8.sum()
    err_blockwise = (O - O_blockwise).abs().max().item()
    
    print(f"\n" + "=" * 60)
    print(f"Kernel vs ROW-WISE:   max_err = {err_rowwise:.6f}")
    print(f"Kernel vs BLOCK-WISE: max_err = {err_blockwise:.6f}")
    
    if err_rowwise < err_blockwise:
        print("✅ Kernel is doing ROW-WISE softmax!")
    else:
        print("❌ Kernel is still BLOCK-WISE")
        
        # Debug: check if output might be transposed
        print("\nChecking if output is transposed...")
        if O.shape[0] == O.shape[1]:
            err_rowwise_T = (O.T - O_rowwise).abs().max().item()
            print(f"Kernel.T vs ROW-WISE: max_err = {err_rowwise_T:.6f}")
    
    hip.hipModuleUnload(module)


if __name__ == "__main__":
    build_and_run()

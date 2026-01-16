#!/usr/bin/env python3
"""
Try different argument layouts for Triton HSACO.

From assembly analysis:
- s[18:19] loaded from offset 0x60 (96)
- s18 used in division (likely H for computing off_h)
- s19 compared with 0 and loop counter (likely N_CTX)

Try: offset 96 = H, offset 100 = N_CTX
"""

import os
import ctypes
import struct
import math
import subprocess

os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch

hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
_ = torch.zeros(1, device='cuda')

def compile_triton_asm():
    src = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s"
    obj = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.co"
    cmd = f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o {obj} {src}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compile error: {result.stderr}")
        return False
    print(f"Compiled {obj}")
    return True

def load_kernel():
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    path = b'/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.co'
    err = hip.hipModuleLoad(ctypes.byref(module), path)
    if err != 0:
        print(f"hipModuleLoad failed: {err}")
        return None, None
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b'_attn_fwd')
    if err != 0:
        print(f"hipModuleGetFunction failed: {err}")
        return None, None
    print("Kernel loaded")
    return module, func


def try_layout(func, layout_name, build_kernarg_func):
    """Try a specific argument layout."""
    print(f"\n{'='*60}")
    print(f"Trying layout: {layout_name}")
    print(f"{'='*60}")
    
    batch, heads, seq_len, head_dim = 1, 1, 128, 128
    
    Q = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    K = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    V = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    Out = torch.zeros(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    M = torch.zeros(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    kernarg = build_kernarg_func(Q, K, V, sm_scale, M, Out, batch, heads, seq_len)
    
    # Allocate on device
    kernarg_ptr = ctypes.c_void_p()
    hip.hipMalloc(ctypes.byref(kernarg_ptr), 256)
    hip.hipMemcpy(kernarg_ptr, (ctypes.c_char * len(kernarg)).from_buffer(kernarg), len(kernarg), 1)
    
    BLOCK_M = 128
    grid_x = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_y = batch * heads
    
    print(f"Grid: ({grid_x}, {grid_y}), Threads: 256")
    
    try:
        err = hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, 24576, None, kernarg_ptr, None)
        if err != 0:
            print(f"Launch error: {err}")
            hip.hipFree(kernarg_ptr)
            return False
        
        err = hip.hipDeviceSynchronize()
        if err != 0:
            print(f"Sync error: {err}")
            hip.hipFree(kernarg_ptr)
            return False
        
        print(f"SUCCESS! Output: [{Out.min().item():.4f}, {Out.max().item():.4f}], mean={Out.mean().item():.4f}")
        hip.hipFree(kernarg_ptr)
        return True
        
    except Exception as e:
        print(f"Exception: {e}")
        hip.hipFree(kernarg_ptr)
        return False


def layout_original(Q, K, V, sm_scale, M, Out, batch, heads, seq_len):
    """Original layout from metadata."""
    kernarg = bytearray(128)
    struct.pack_into('Q', kernarg, 0, Q.data_ptr())
    struct.pack_into('Q', kernarg, 8, K.data_ptr())
    struct.pack_into('Q', kernarg, 16, V.data_ptr())
    struct.pack_into('f', kernarg, 24, sm_scale)
    struct.pack_into('Q', kernarg, 32, M.data_ptr())
    struct.pack_into('Q', kernarg, 40, Out.data_ptr())
    struct.pack_into('i', kernarg, 48, Q.stride(0))
    struct.pack_into('i', kernarg, 52, Q.stride(1))
    struct.pack_into('i', kernarg, 56, Q.stride(2))
    struct.pack_into('i', kernarg, 60, Q.stride(3))
    struct.pack_into('i', kernarg, 64, K.stride(0))
    struct.pack_into('i', kernarg, 68, K.stride(1))
    struct.pack_into('i', kernarg, 72, K.stride(2))
    struct.pack_into('i', kernarg, 76, K.stride(3))
    struct.pack_into('i', kernarg, 80, V.stride(0))
    struct.pack_into('i', kernarg, 84, V.stride(1))
    struct.pack_into('i', kernarg, 88, V.stride(2))
    struct.pack_into('i', kernarg, 92, V.stride(3))
    struct.pack_into('i', kernarg, 96, Out.stride(0))
    struct.pack_into('i', kernarg, 100, Out.stride(1))
    struct.pack_into('i', kernarg, 104, Out.stride(2))
    struct.pack_into('i', kernarg, 108, Out.stride(3))
    struct.pack_into('i', kernarg, 112, batch)
    struct.pack_into('i', kernarg, 116, heads)
    return kernarg


def layout_h_nctx_at_96(Q, K, V, sm_scale, M, Out, batch, heads, seq_len):
    """H and N_CTX at offset 96, 100 (based on s18/s19 usage)."""
    kernarg = bytearray(128)
    struct.pack_into('Q', kernarg, 0, Q.data_ptr())
    struct.pack_into('Q', kernarg, 8, K.data_ptr())
    struct.pack_into('Q', kernarg, 16, V.data_ptr())
    struct.pack_into('f', kernarg, 24, sm_scale)
    struct.pack_into('Q', kernarg, 32, M.data_ptr())
    struct.pack_into('Q', kernarg, 40, Out.data_ptr())
    struct.pack_into('i', kernarg, 48, Q.stride(0))
    struct.pack_into('i', kernarg, 52, Q.stride(1))
    struct.pack_into('i', kernarg, 56, Q.stride(2))
    struct.pack_into('i', kernarg, 60, Q.stride(3))
    struct.pack_into('i', kernarg, 64, K.stride(0))
    struct.pack_into('i', kernarg, 68, K.stride(1))
    struct.pack_into('i', kernarg, 72, K.stride(2))
    struct.pack_into('i', kernarg, 76, K.stride(3))
    struct.pack_into('i', kernarg, 80, V.stride(0))
    struct.pack_into('i', kernarg, 84, V.stride(1))
    struct.pack_into('i', kernarg, 88, V.stride(2))
    struct.pack_into('i', kernarg, 92, V.stride(3))
    # Put H at 96, N_CTX at 100
    struct.pack_into('i', kernarg, 96, heads)
    struct.pack_into('i', kernarg, 100, seq_len)
    struct.pack_into('i', kernarg, 104, Out.stride(0))
    struct.pack_into('i', kernarg, 108, Out.stride(1))
    struct.pack_into('i', kernarg, 112, Out.stride(2))
    struct.pack_into('i', kernarg, 116, Out.stride(3))
    return kernarg


def layout_z_h_at_96(Q, K, V, sm_scale, M, Out, batch, heads, seq_len):
    """Z and H at offset 96, 100."""
    kernarg = bytearray(128)
    struct.pack_into('Q', kernarg, 0, Q.data_ptr())
    struct.pack_into('Q', kernarg, 8, K.data_ptr())
    struct.pack_into('Q', kernarg, 16, V.data_ptr())
    struct.pack_into('f', kernarg, 24, sm_scale)
    struct.pack_into('Q', kernarg, 32, M.data_ptr())
    struct.pack_into('Q', kernarg, 40, Out.data_ptr())
    struct.pack_into('i', kernarg, 48, Q.stride(0))
    struct.pack_into('i', kernarg, 52, Q.stride(1))
    struct.pack_into('i', kernarg, 56, Q.stride(2))
    struct.pack_into('i', kernarg, 60, Q.stride(3))
    struct.pack_into('i', kernarg, 64, K.stride(0))
    struct.pack_into('i', kernarg, 68, K.stride(1))
    struct.pack_into('i', kernarg, 72, K.stride(2))
    struct.pack_into('i', kernarg, 76, K.stride(3))
    struct.pack_into('i', kernarg, 80, V.stride(0))
    struct.pack_into('i', kernarg, 84, V.stride(1))
    struct.pack_into('i', kernarg, 88, V.stride(2))
    struct.pack_into('i', kernarg, 92, V.stride(3))
    # Put Z at 96, H at 100
    struct.pack_into('i', kernarg, 96, batch)
    struct.pack_into('i', kernarg, 100, heads)
    struct.pack_into('i', kernarg, 104, seq_len)
    struct.pack_into('i', kernarg, 108, Out.stride(0))
    struct.pack_into('i', kernarg, 112, Out.stride(1))
    struct.pack_into('i', kernarg, 116, Out.stride(2))
    return kernarg


def layout_exact_python_order(Q, K, V, sm_scale, M, Out, batch, heads, seq_len):
    """Exact order from Python: strides then Z, H, N_CTX."""
    kernarg = bytearray(128)
    offset = 0
    
    # Q, K, V, sm_scale, M, Out
    struct.pack_into('Q', kernarg, offset, Q.data_ptr()); offset += 8
    struct.pack_into('Q', kernarg, offset, K.data_ptr()); offset += 8
    struct.pack_into('Q', kernarg, offset, V.data_ptr()); offset += 8
    struct.pack_into('f', kernarg, offset, sm_scale); offset += 4
    offset += 4  # padding
    struct.pack_into('Q', kernarg, offset, M.data_ptr()); offset += 8
    struct.pack_into('Q', kernarg, offset, Out.data_ptr()); offset += 8
    
    # Q strides
    for s in Q.stride(): struct.pack_into('i', kernarg, offset, s); offset += 4
    # K strides
    for s in K.stride(): struct.pack_into('i', kernarg, offset, s); offset += 4
    # V strides
    for s in V.stride(): struct.pack_into('i', kernarg, offset, s); offset += 4
    # O strides
    for s in Out.stride(): struct.pack_into('i', kernarg, offset, s); offset += 4
    
    # Z, H, N_CTX
    struct.pack_into('i', kernarg, offset, batch); offset += 4
    struct.pack_into('i', kernarg, offset, heads); offset += 4
    struct.pack_into('i', kernarg, offset, seq_len); offset += 4
    
    print(f"Final offset: {offset} (kernarg_size=120)")
    return kernarg


if __name__ == "__main__":
    if not compile_triton_asm():
        exit(1)
    
    module, func = load_kernel()
    if func is None:
        exit(1)
    
    layouts = [
        ("original", layout_original),
        ("h_nctx_at_96", layout_h_nctx_at_96),
        ("z_h_at_96", layout_z_h_at_96),
        ("exact_python_order", layout_exact_python_order),
    ]
    
    for name, builder in layouts:
        try_layout(func, name, builder)
    
    hip.hipModuleUnload(module)

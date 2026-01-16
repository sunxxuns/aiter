#!/usr/bin/env python3
"""
Final attempt: match exact shapes from assembly dump.
The assembly shows it was dumped with seq_len=1024.
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

def compile_and_load():
    src = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s"
    obj = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.co"
    subprocess.run(f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o {obj} {src}", 
                   shell=True, check=True)
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), obj.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, b'_attn_fwd')
    return module, func


def test_with_1024(func):
    """Try with seq_len=1024 (what the assembly was compiled for)."""
    print("\n" + "="*60)
    print("Test with seq_len=1024 (matches assembly dump)")
    print("="*60)
    
    batch, heads, seq_len, head_dim = 1, 1, 1024, 128
    
    Q = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    K = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    V = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    Out = torch.zeros(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    M = torch.zeros(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    print(f"Shapes: Q={Q.shape}, Out={Out.shape}")
    print(f"Strides: Q={Q.stride()}, Out={Out.stride()}")
    
    # Build kernarg - exactly 120 bytes based on metadata
    kernarg = bytearray(128)
    
    struct.pack_into('Q', kernarg, 0, Q.data_ptr())      # Q
    struct.pack_into('Q', kernarg, 8, K.data_ptr())      # K
    struct.pack_into('Q', kernarg, 16, V.data_ptr())     # V
    struct.pack_into('f', kernarg, 24, sm_scale)         # sm_scale
    # 28-31: padding
    struct.pack_into('Q', kernarg, 32, M.data_ptr())     # M
    struct.pack_into('Q', kernarg, 40, Out.data_ptr())   # Out
    
    # Strides
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
    struct.pack_into('i', kernarg, 112, batch)           # Z
    struct.pack_into('i', kernarg, 116, heads)           # H
    # Note: N_CTX at 120 would be outside kernarg_size=120
    # It might be baked in as constexpr
    
    kernarg_ptr = ctypes.c_void_p()
    hip.hipMalloc(ctypes.byref(kernarg_ptr), 256)
    hip.hipMemcpy(kernarg_ptr, (ctypes.c_char * len(kernarg)).from_buffer(kernarg), len(kernarg), 1)
    
    BLOCK_M = 128
    grid_x = (seq_len + BLOCK_M - 1) // BLOCK_M  # 8 blocks
    grid_y = batch * heads  # 1
    
    print(f"Grid: ({grid_x}, {grid_y})")
    
    # Try different LDS sizes
    for lds_size in [0, 16384, 24576, 32768, 65536]:
        Out.zero_()
        M.zero_()
        
        print(f"\n  LDS={lds_size}...")
        try:
            err = hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, lds_size, None, kernarg_ptr, None)
            if err != 0:
                print(f"    Launch error: {err}")
                continue
            
            err = hip.hipDeviceSynchronize()
            if err != 0:
                print(f"    Sync error: {err}")
                continue
            
            print(f"    SUCCESS! Out=[{Out.min().item():.2f}, {Out.max().item():.2f}], mean={Out.mean().item():.4f}")
            break
            
        except Exception as e:
            print(f"    Exception: {e}")
    
    hip.hipFree(kernarg_ptr)


if __name__ == "__main__":
    try:
        module, func = compile_and_load()
        print("Kernel loaded successfully")
        test_with_1024(func)
        hip.hipModuleUnload(module)
    except Exception as e:
        print(f"Failed: {e}")

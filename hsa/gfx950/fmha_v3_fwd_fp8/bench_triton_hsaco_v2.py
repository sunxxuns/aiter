#!/usr/bin/env python3
"""
Benchmark Triton's compiled FP8 flash attention HSACO directly.

From metadata analysis (120 bytes):
- offset 0:   Q ptr (8)
- offset 8:   K ptr (8)  
- offset 16:  V ptr (8)
- offset 24:  sm_scale (4) + padding (4)
- offset 32:  M ptr (8)
- offset 40:  Out ptr (8)
- offset 48-100: 14 int32 strides
- offset 104: mystery ptr1 (8) - maybe stride_om/on as 64-bit?
- offset 112: mystery ptr2 (8) - maybe Z/H as 64-bit?
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
    """Compile the Triton assembly to HSACO."""
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
    """Load the Triton kernel."""
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
        
    print("Kernel loaded successfully")
    return module, func


def test_small(func):
    """Test with small tensors first."""
    print("\n" + "="*70)
    print("Small Test: B=1, H=1, S=128, D=128")
    print("="*70)
    
    batch, heads, seq_len, head_dim = 1, 1, 128, 128
    
    # Create tensors - contiguous, simple layout
    Q = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    K = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    V = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    Out = torch.zeros(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    M = torch.zeros(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    print(f"Tensor shapes: Q={Q.shape}, K={K.shape}, V={V.shape}, Out={Out.shape}")
    print(f"Strides: Q={Q.stride()}, K={K.stride()}, V={V.stride()}, Out={Out.stride()}")
    print(f"sm_scale = {sm_scale}")
    
    # Grid
    BLOCK_M = 128
    grid_x = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_y = batch * heads
    print(f"Grid: ({grid_x}, {grid_y}, 1)")
    
    # Build kernarg buffer (120 bytes)
    kernarg = bytearray(128)  # Extra padding
    
    # Pointers
    struct.pack_into('Q', kernarg, 0, Q.data_ptr())
    struct.pack_into('Q', kernarg, 8, K.data_ptr())
    struct.pack_into('Q', kernarg, 16, V.data_ptr())
    struct.pack_into('f', kernarg, 24, sm_scale)
    struct.pack_into('I', kernarg, 28, 0)  # padding
    struct.pack_into('Q', kernarg, 32, M.data_ptr())
    struct.pack_into('Q', kernarg, 40, Out.data_ptr())
    
    # Strides (in elements)
    struct.pack_into('i', kernarg, 48, Q.stride(0))   # stride_qz
    struct.pack_into('i', kernarg, 52, Q.stride(1))   # stride_qh
    struct.pack_into('i', kernarg, 56, Q.stride(2))   # stride_qm
    struct.pack_into('i', kernarg, 60, Q.stride(3))   # stride_qk
    struct.pack_into('i', kernarg, 64, K.stride(0))   # stride_kz
    struct.pack_into('i', kernarg, 68, K.stride(1))   # stride_kh
    struct.pack_into('i', kernarg, 72, K.stride(2))   # stride_kn
    struct.pack_into('i', kernarg, 76, K.stride(3))   # stride_kk
    struct.pack_into('i', kernarg, 80, V.stride(0))   # stride_vz
    struct.pack_into('i', kernarg, 84, V.stride(1))   # stride_vh
    struct.pack_into('i', kernarg, 88, V.stride(2))   # stride_vk
    struct.pack_into('i', kernarg, 92, V.stride(3))   # stride_vn
    struct.pack_into('i', kernarg, 96, Out.stride(0)) # stride_oz
    struct.pack_into('i', kernarg, 100, Out.stride(1))# stride_oh
    
    # Mystery bytes at 104 and 112 - try stride_om/on + Z/H/N_CTX
    struct.pack_into('i', kernarg, 104, Out.stride(2))  # stride_om
    struct.pack_into('i', kernarg, 108, Out.stride(3))  # stride_on
    struct.pack_into('i', kernarg, 112, batch)          # Z
    struct.pack_into('i', kernarg, 116, heads)          # H
    # N_CTX might be constexpr (baked in), but let's try
    
    print(f"\nKernarg bytes 104-120: {kernarg[104:120].hex()}")
    
    # Allocate kernarg on device
    kernarg_ptr = ctypes.c_void_p()
    hip.hipMalloc(ctypes.byref(kernarg_ptr), 256)
    hip.hipMemcpy(kernarg_ptr, (ctypes.c_char * len(kernarg)).from_buffer(kernarg), 
                  len(kernarg), 1)
    
    # Try to launch
    print("\nLaunching kernel...")
    try:
        # LDS size from metadata: group_segment_fixed_size = 0
        # But Triton uses dynamic LDS, let's try 24KB
        lds_size = 24576
        
        err = hip.hipModuleLaunchKernel(
            func,
            grid_x, grid_y, 1,
            256, 1, 1,  # 256 threads
            lds_size,
            None,
            kernarg_ptr,
            None
        )
        
        if err != 0:
            print(f"hipModuleLaunchKernel returned: {err}")
        
        err = hip.hipDeviceSynchronize()
        if err != 0:
            print(f"hipDeviceSynchronize returned: {err}")
        else:
            print("Kernel completed!")
            print(f"Output range: [{Out.min().item():.4f}, {Out.max().item():.4f}]")
            print(f"Output mean: {Out.mean().item():.4f}")
            print(f"M range: [{M.min().item():.4f}, {M.max().item():.4f}]")
            
    except Exception as e:
        print(f"Exception: {e}")
    
    hip.hipFree(kernarg_ptr)


def test_with_explicit_args(func):
    """Try with explicitly allocated argument buffer matching Triton's layout."""
    print("\n" + "="*70)
    print("Test with ctypes argument array")
    print("="*70)
    
    batch, heads, seq_len, head_dim = 1, 1, 128, 128
    
    Q = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    K = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    V = torch.ones(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    Out = torch.zeros(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    M = torch.zeros(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Build argument array like hipModuleLaunchKernel expects
    # Each argument is passed as a pointer to its value
    
    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ('Q', ctypes.c_void_p),
            ('K', ctypes.c_void_p),
            ('V', ctypes.c_void_p),
            ('sm_scale', ctypes.c_float),
            ('_pad1', ctypes.c_uint32),
            ('M', ctypes.c_void_p),
            ('Out', ctypes.c_void_p),
            ('stride_qz', ctypes.c_int32),
            ('stride_qh', ctypes.c_int32),
            ('stride_qm', ctypes.c_int32),
            ('stride_qk', ctypes.c_int32),
            ('stride_kz', ctypes.c_int32),
            ('stride_kh', ctypes.c_int32),
            ('stride_kn', ctypes.c_int32),
            ('stride_kk', ctypes.c_int32),
            ('stride_vz', ctypes.c_int32),
            ('stride_vh', ctypes.c_int32),
            ('stride_vk', ctypes.c_int32),
            ('stride_vn', ctypes.c_int32),
            ('stride_oz', ctypes.c_int32),
            ('stride_oh', ctypes.c_int32),
            ('stride_om', ctypes.c_int32),
            ('stride_on', ctypes.c_int32),
            ('Z', ctypes.c_int32),
            ('H', ctypes.c_int32),
        ]
    
    args = KernelArgs()
    args.Q = Q.data_ptr()
    args.K = K.data_ptr()
    args.V = V.data_ptr()
    args.sm_scale = sm_scale
    args._pad1 = 0
    args.M = M.data_ptr()
    args.Out = Out.data_ptr()
    args.stride_qz = Q.stride(0)
    args.stride_qh = Q.stride(1)
    args.stride_qm = Q.stride(2)
    args.stride_qk = Q.stride(3)
    args.stride_kz = K.stride(0)
    args.stride_kh = K.stride(1)
    args.stride_kn = K.stride(2)
    args.stride_kk = K.stride(3)
    args.stride_vz = V.stride(0)
    args.stride_vh = V.stride(1)
    args.stride_vk = V.stride(2)
    args.stride_vn = V.stride(3)
    args.stride_oz = Out.stride(0)
    args.stride_oh = Out.stride(1)
    args.stride_om = Out.stride(2)
    args.stride_on = Out.stride(3)
    args.Z = batch
    args.H = heads
    
    print(f"Struct size: {ctypes.sizeof(args)} bytes")
    
    # Allocate on device
    kernarg_ptr = ctypes.c_void_p()
    hip.hipMalloc(ctypes.byref(kernarg_ptr), 256)
    hip.hipMemcpy(kernarg_ptr, ctypes.byref(args), ctypes.sizeof(args), 1)
    
    BLOCK_M = 128
    grid_x = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_y = batch * heads
    
    print(f"Grid: ({grid_x}, {grid_y}, 1)")
    print("Launching...")
    
    try:
        err = hip.hipModuleLaunchKernel(
            func, grid_x, grid_y, 1,
            256, 1, 1, 24576, None, kernarg_ptr, None
        )
        print(f"Launch returned: {err}")
        
        err = hip.hipDeviceSynchronize()
        print(f"Sync returned: {err}")
        
        if err == 0:
            print(f"Output: [{Out.min().item():.4f}, {Out.max().item():.4f}]")
            
    except Exception as e:
        print(f"Exception: {e}")
    
    hip.hipFree(kernarg_ptr)


if __name__ == "__main__":
    if not compile_triton_asm():
        exit(1)
    
    module, func = load_kernel()
    if func is None:
        exit(1)
    
    test_small(func)
    test_with_explicit_args(func)
    
    hip.hipModuleUnload(module)

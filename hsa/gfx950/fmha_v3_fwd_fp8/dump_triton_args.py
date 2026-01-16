#!/usr/bin/env python3
"""
Dump the actual arguments Triton passes to the kernel.
"""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

import torch
import triton
import triton.language as tl
import math

@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # Simple kernel that just writes zeros
    tl.store(Out + off_hz * stride_oh + start_m * BLOCK_M * stride_om, 0.0)


def main():
    batch, heads, seq_len, head_dim = 1, 1, 128, 128
    
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    o = torch.zeros(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    m = torch.zeros(batch, heads, seq_len, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    print(f"Q ptr: {hex(q.data_ptr())}")
    print(f"K ptr: {hex(k.data_ptr())}")
    print(f"V ptr: {hex(v.data_ptr())}")
    print(f"sm_scale: {sm_scale}")
    print(f"M ptr: {hex(m.data_ptr())}")
    print(f"Out ptr: {hex(o.data_ptr())}")
    print()
    print(f"Q strides: {q.stride()}")
    print(f"K strides: {k.stride()}")
    print(f"V strides: {v.stride()}")
    print(f"O strides: {o.stride()}")
    print()
    print(f"Z={batch}, H={heads}, N_CTX={seq_len}")
    
    # Calculate expected kernarg bytes
    print("\n=== Expected Kernarg Layout (120 bytes) ===")
    offset = 0
    print(f"offset {offset}: Q ptr = {hex(q.data_ptr())}"); offset += 8
    print(f"offset {offset}: K ptr = {hex(k.data_ptr())}"); offset += 8
    print(f"offset {offset}: V ptr = {hex(v.data_ptr())}"); offset += 8
    print(f"offset {offset}: sm_scale = {sm_scale}"); offset += 4
    print(f"offset {offset}: (padding)"); offset += 4
    print(f"offset {offset}: M ptr = {hex(m.data_ptr())}"); offset += 8
    print(f"offset {offset}: Out ptr = {hex(o.data_ptr())}"); offset += 8
    
    print(f"offset {offset}: stride_qz = {q.stride(0)}"); offset += 4
    print(f"offset {offset}: stride_qh = {q.stride(1)}"); offset += 4
    print(f"offset {offset}: stride_qm = {q.stride(2)}"); offset += 4
    print(f"offset {offset}: stride_qk = {q.stride(3)}"); offset += 4
    
    print(f"offset {offset}: stride_kz = {k.stride(0)}"); offset += 4
    print(f"offset {offset}: stride_kh = {k.stride(1)}"); offset += 4
    print(f"offset {offset}: stride_kn = {k.stride(2)}"); offset += 4
    print(f"offset {offset}: stride_kk = {k.stride(3)}"); offset += 4
    
    print(f"offset {offset}: stride_vz = {v.stride(0)}"); offset += 4
    print(f"offset {offset}: stride_vh = {v.stride(1)}"); offset += 4
    print(f"offset {offset}: stride_vk = {v.stride(2)}"); offset += 4
    print(f"offset {offset}: stride_vn = {v.stride(3)}"); offset += 4
    
    print(f"offset {offset}: stride_oz = {o.stride(0)}"); offset += 4
    print(f"offset {offset}: stride_oh = {o.stride(1)}"); offset += 4
    print(f"offset {offset}: stride_om = {o.stride(2)}"); offset += 4
    print(f"offset {offset}: stride_on = {o.stride(3)}"); offset += 4
    
    print(f"offset {offset}: Z = {batch}"); offset += 4
    print(f"offset {offset}: H = {heads}"); offset += 4
    print(f"offset {offset}: N_CTX = {seq_len}"); offset += 4
    
    print(f"\nTotal: {offset} bytes")
    
    # Grid calculation
    BLOCK_M = 128
    grid_x = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_y = batch * heads
    print(f"\nGrid: ({grid_x}, {grid_y}, 1)")


if __name__ == "__main__":
    main()

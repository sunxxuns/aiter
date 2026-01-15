#!/usr/bin/env python3
"""Dump Triton FP8 flash attention assembly."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache_fmha'

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
                                    strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
                                    strides=(stride_vk, stride_vn), offsets=(0, 0),
                                    block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
                                    strides=(stride_kk, stride_kn), offsets=(0, 0),
                                    block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
                                    strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    
    q = tl.load(Q_block_ptr)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def main():
    batch, heads, seq_len, head_dim = 1, 1, 1024, 128
    
    # FP8 tensors
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    o = torch.empty(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_M']), batch * heads, 1)
    
    print("Compiling FP8 flash attention...")
    
    # Compile with specific config
    compiled = _attn_fwd[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, heads, seq_len,
        HEAD_DIM=head_dim, BLOCK_M=128, BLOCK_N=64,
        num_warps=4, num_stages=1,
        waves_per_eu=2, allow_flush_denorm=True,
    )
    
    # Get assembly
    asm = compiled.asm.get('amdgcn', None)
    if asm:
        asm_file = "/tmp/triton_fmha_fp8_gfx950.s"
        with open(asm_file, 'w') as f:
            f.write(asm)
        print(f"Assembly saved to: {asm_file}")
        print(f"Size: {len(asm)} bytes, {len(asm.split(chr(10)))} lines")
        
        # Analyze
        lines = asm.split('\n')
        
        mfma = [l for l in lines if 'mfma' in l.lower()]
        ds_read = [l for l in lines if 'ds_read' in l.lower() or 'ds_load' in l.lower()]
        ds_write = [l for l in lines if 'ds_write' in l.lower() or 'ds_store' in l.lower()]
        global_load = [l for l in lines if 'global_load' in l.lower()]
        buffer_load = [l for l in lines if 'buffer_load' in l.lower()]
        exp2 = [l for l in lines if 'v_exp_' in l.lower() or 'exp2' in l.lower()]
        
        print(f"\nInstruction counts:")
        print(f"  MFMA:         {len(mfma)}")
        print(f"  ds_read:      {len(ds_read)}")
        print(f"  ds_write:     {len(ds_write)}")
        print(f"  global_load:  {len(global_load)}")
        print(f"  buffer_load:  {len(buffer_load)}")
        print(f"  exp2:         {len(exp2)}")
        
        print(f"\nMFMA instructions:")
        for m in mfma[:10]:
            print(f"  {m.strip()}")
        
        print(f"\nDS read patterns (first 15):")
        for d in ds_read[:15]:
            print(f"  {d.strip()}")
            
    else:
        print(f"No assembly available. Keys: {list(compiled.asm.keys())}")


if __name__ == "__main__":
    main()

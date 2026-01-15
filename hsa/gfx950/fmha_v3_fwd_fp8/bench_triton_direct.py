#!/usr/bin/env python3
"""Direct FP8 vs FP16 Triton flash attention comparison."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

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


def benchmark(dtype, batch, heads, seq_len, head_dim, warmup=10, iters=100):
    if dtype == torch.float8_e4m3fn:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
        o = torch.empty(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    else:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
        o = torch.empty(batch, heads, seq_len, head_dim, device='cuda', dtype=dtype)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_M']), batch * heads, 1)
    
    # Warmup
    for _ in range(warmup):
        _attn_fwd[grid](
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
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        _attn_fwd[grid](
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
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iters
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    
    return elapsed_ms, tflops


def main():
    print("=" * 80)
    print("Triton Flash Attention: FP8 vs FP16 vs BF16")
    print("=" * 80)
    
    configs = [
        (1, 40, 4096, 128),
        (1, 40, 8192, 128),
        (1, 40, 16384, 128),
        (1, 1, 32130, 128),
        (1, 40, 32130, 128),
    ]
    
    print(f"\n{'Config':<30} {'FP8 TF/s':<12} {'FP16 TF/s':<12} {'BF16 TF/s':<12} {'FP8/FP16':<10}")
    print("-" * 86)
    
    for batch, heads, seq_len, head_dim in configs:
        config_str = f"B={batch}, H={heads}, S={seq_len}"
        
        try:
            fp8_ms, fp8_tf = benchmark(torch.float8_e4m3fn, batch, heads, seq_len, head_dim)
            fp16_ms, fp16_tf = benchmark(torch.float16, batch, heads, seq_len, head_dim)
            bf16_ms, bf16_tf = benchmark(torch.bfloat16, batch, heads, seq_len, head_dim)
            
            speedup = fp8_tf / fp16_tf
            print(f"{config_str:<30} {fp8_tf:<12.1f} {fp16_tf:<12.1f} {bf16_tf:<12.1f} {speedup:<10.2f}x")
        except Exception as e:
            print(f"{config_str:<30} ERROR: {e}")


if __name__ == "__main__":
    main()

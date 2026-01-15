#!/usr/bin/env python3
"""
Compile Triton FP8 attention for gfx950 and dump the assembly.
"""

import os
import torch
import triton
import triton.language as tl

os.environ['HIP_VISIBLE_DEVICES'] = '0'

# Simple FP8 QK matmul kernel similar to our use case
@triton.jit
def fp8_qk_kernel(
    Q_ptr, K_ptr, O_ptr,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_om, stride_on,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Simple FP8 Q@K^T kernel."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K loop
    for k in range(0, K, BLOCK_K):
        q = tl.load(q_ptrs)
        k_tile = tl.load(k_ptrs)
        
        # FP8 dot product
        acc += tl.dot(q, k_tile.T)
        
        q_ptrs += BLOCK_K * stride_qk
        k_ptrs += BLOCK_K * stride_kk
    
    # Store
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, acc)


def main():
    print("=" * 70)
    print("Compiling Triton FP8 QK kernel for gfx950")
    print("=" * 70)
    
    # Setup
    M, N, K = 32, 64, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    
    # Check if FP8 is available
    if not hasattr(torch, 'float8_e4m3fn'):
        print("FP8 not available in this PyTorch version")
        return
    
    # Create tensors
    Q = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    K_mat = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
    O = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    
    # Grid
    grid = (M // BLOCK_M, N // BLOCK_N)
    
    # Compile and get assembly
    print("\nCompiling kernel...")
    compiled = fp8_qk_kernel[grid](
        Q, K_mat, O,
        Q.stride(0), Q.stride(1),
        K_mat.stride(0), K_mat.stride(1),
        O.stride(0), O.stride(1),
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    # Get the ASM
    asm = compiled.asm.get('amdgcn', None)
    if asm:
        # Save to file
        asm_file = "/tmp/triton_fp8_qk_gfx950.s"
        with open(asm_file, 'w') as f:
            f.write(asm)
        print(f"\nAssembly saved to: {asm_file}")
        print(f"Size: {len(asm)} bytes")
        
        # Show key sections
        print("\n--- KEY PATTERNS IN ASSEMBLY ---")
        lines = asm.split('\n')
        
        # Find ds_read patterns
        ds_patterns = [l for l in lines if 'ds_read' in l.lower() or 'ds_load' in l.lower()]
        print(f"\nDS read patterns ({len(ds_patterns)} occurrences):")
        for p in ds_patterns[:10]:
            print(f"  {p.strip()}")
        
        # Find mfma patterns
        mfma_patterns = [l for l in lines if 'mfma' in l.lower()]
        print(f"\nMFMA patterns ({len(mfma_patterns)} occurrences):")
        for p in mfma_patterns[:5]:
            print(f"  {p.strip()}")
        
        # Find LDS write patterns
        ds_write_patterns = [l for l in lines if 'ds_write' in l.lower() or 'ds_store' in l.lower()]
        print(f"\nDS write patterns ({len(ds_write_patterns)} occurrences):")
        for p in ds_write_patterns[:10]:
            print(f"  {p.strip()}")
        
        # Show buffer load patterns
        buf_patterns = [l for l in lines if 'buffer_load' in l.lower()]
        print(f"\nBuffer load patterns ({len(buf_patterns)} occurrences):")
        for p in buf_patterns[:10]:
            print(f"  {p.strip()}")
            
    else:
        print("Could not get assembly")
        print(f"Available keys: {list(compiled.asm.keys())}")

if __name__ == "__main__":
    main()

# FP8 Flash Attention Forward Kernel Design

## Overview

This document describes the design for an FP8 flash attention forward kernel targeting gfx950 (MI300X/MI350). The goal is to achieve >30% speedup over the BF16 baseline (~1000 TF/s → >1300 TF/s).

## Reference Kernels

- **BF16 FMHA**: `/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16.co`
- **FP8 GEMM**: `/sgl-workspace/aiter/hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale_BpreShuffle_128x128.co`

## Key Differences: BF16 vs FP8

### MFMA Instructions

| Aspect | BF16 | FP8 |
|--------|------|-----|
| Instruction | `v_mfma_f32_32x32x16_bf16` | `v_mfma_f32_32x32x16_fp8_fp8` |
| Input A regs | 4 VGPRs (128-bit) | 2 AGPRs (64-bit) |
| Input B regs | 4 VGPRs (128-bit) | 2 VGPRs (64-bit) |
| Output regs | 16 VGPRs | 16 VGPRs |
| K per inst | 16 | 16 |
| Elements/reg | 8 BF16 | 8 FP8 |

### Memory and Register Efficiency

- **FP8**: 1 byte/element vs BF16: 2 bytes/element
- Same memory load (e.g., `buffer_load_dwordx4`) brings 2x more FP8 elements
- Half the registers needed for same number of elements
- Potential for 2x more data in LDS, or same data in half LDS space

## Kernel Arguments

BF16 kernel arguments (from metadata):
```
Offset  Size  Name
0x00    8     ptr_R (output)
0x10    8     ptr_Q
0x20    8     ptr_K
0x30    8     ptr_V
0x40    8     ptr_LSE
0x50    4     softmax_scale
0x60    4     seq_len_q
...     ...   strides, etc.
```

FP8 kernel needs additional scale factors:
```
0x200   4     q_scale (FP32)
0x204   4     k_scale (FP32)
0x208   4     v_scale (FP32)
```

## Algorithm

Same as BF16 Flash Attention, with scaling:

```
For each K/V tile:
    # QK GEMM (FP8 inputs)
    S = mfma_fp8(Q_fp8, K_fp8^T)  # Result in FP32
    S = S * (q_scale * k_scale * softmax_scale)
    
    # Online Softmax (FP32)
    m_new = max(m_old, rowmax(S))
    P = exp(S - m_new)
    l_new = l_old * exp(m_old - m_new) + rowsum(P)
    O = O * exp(m_old - m_new)
    
    # PV GEMM (FP8 V, FP32 P → need to quantize P to FP8)
    # Option 1: Keep P in FP32, convert V to FP32, use FP32 FMA
    # Option 2: Quantize P to FP8, use FP8 MFMA (need runtime scale)
    O += P @ V  # or mfma_fp8(P_fp8, V_fp8)
    
# Final output
O = O / l * v_scale  # Apply V scale
Output = cvt_bf16(O)  # Convert FP32 to BF16
```

## Key Design Decisions

### 1. P Quantization for PV GEMM

**Option A: FP32 P × FP8 V (simpler, likely baseline)**
- Keep P in FP32 after softmax
- Load V as FP8, upcast to FP32 for FMA
- Use VOP3 `v_fma_f32` instructions
- Pro: No dynamic quantization needed
- Con: Doesn't use FP8 MFMA for PV, lower throughput

**Option B: FP8 P × FP8 V (optimal)**
- After softmax, quantize P to FP8 with dynamic scale
- Use `v_mfma_f32_32x32x16_fp8_fp8` for PV GEMM
- Pro: Full FP8 MFMA utilization
- Con: Need `v_cvt_pk_fp8_f32` and dynamic scale tracking
- Numerical accuracy concerns

**Decision: Start with Option A, optimize to Option B if needed**

### 2. LDS Layout

BF16 kernel uses 160KB LDS. For FP8:
- Same tile sizes → 80KB LDS (half the bytes)
- Or double tile sizes → same LDS, 2x more work/tile

**Decision: Keep same tile sizes, use freed LDS for double-buffering**

### 3. Data Loading

BF16: `buffer_load_dwordx4` loads 8 BF16 values
FP8: `buffer_load_dwordx4` loads 16 FP8 values

Can load same bytes, get 2x elements, reducing memory pressure.

## Resource Requirements (Estimated)

| Resource | BF16 | FP8 (Est.) |
|----------|------|------------|
| VGPRs | 256 | 192-224 |
| SGPRs | 96 | 100 |
| LDS | 160KB | 80-100KB |
| Workgroup | 512 threads | 512 threads |

## Performance Analysis

### BF16 Baseline
- 176 MFMA instructions per main loop iteration
- ~1000 TF/s measured

### FP8 Target
- ~88-100 MFMA instructions (FP8 MFMA has 2x K per input register)
- Memory bandwidth: 2x more elements per load
- Target: >1300 TF/s (30%+ improvement)

### Speedup Sources
1. **2x register efficiency**: Half the registers for same data
2. **2x memory efficiency**: Same loads bring 2x elements
3. **Same MFMA throughput**: FP8 MFMA has same cycles as BF16

## File Structure

```
/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/
├── design.md           # This document
├── fwd_hd128_fp8.s     # Assembly source
├── fwd_hd128_fp8.co    # Compiled kernel
├── build.sh            # Build script
└── test.py             # Test/benchmark script
```

## Next Steps

1. Write the assembly kernel following BF16 structure
2. Replace MFMA instructions with FP8 variants
3. Adjust memory loads for FP8 data
4. Add scale factor loading and application
5. Test numerical correctness
6. Benchmark and optimize

# TR8 Layout Analysis for FP8 MFMA

## Problem Statement
`ds_read_b64_tr_b8` (TR8) reads 8 bytes (8 FP8 values) for MFMA operands.
Current kernel uses buffer_load swizzle + TR8, but produces wrong results.
Need to understand what LDS layout TR8 expects.

## TR8 Behavior (from earlier tests)
```
TR8 reads from LDS[base], LDS[base+8], LDS[base+16], LDS[base+24], ...
- Stride = 8 bytes
- Reads 8 elements total = 64 bytes span
```

## MFMA FP8 Requirements
`v_mfma_f32_32x32x16_fp8_fp8`:
- Computes C[32×32] += A[32×16] × B[16×32]
- A operand: 2 VGPRs = 64 bits = 8 FP8 values
- B operand: 2 VGPRs = 64 bits = 8 FP8 values
- k=16 uses two groups of 8 (hence 8 MFMA calls for full 128-dim)

## The Layout Problem

### Row-Major Layout (doesn't work with TR8):
```
Q[32×128] stored as:
  Q[r,k] at LDS[r*128 + k]
  
Row r's first 8 k values at: r*128+0, r*128+1, r*128+2, ..., r*128+7
These are CONSECUTIVE, but TR8 reads with stride 8!
```

### What TR8 Expects:
For TR8 at base `b` to read Q[row, k=0..7]:
```
LDS[b+0]  -> Q[row, 0]
LDS[b+8]  -> Q[row, 1]
LDS[b+16] -> Q[row, 2]
LDS[b+24] -> Q[row, 3]
LDS[b+32] -> Q[row, 4]
LDS[b+40] -> Q[row, 5]
LDS[b+48] -> Q[row, 6]
LDS[b+56] -> Q[row, 7]
```

This requires **column-interleaved** storage!

## Required LDS Layout for TR8

### Option 1: Column-Major with Stride
Store Q[r,k] at: `LDS[r + k*PITCH]` where PITCH >= 32 (number of rows)

For Q[32×128]:
```
Q[r,k] at LDS[r + k*32]

Then TR8 with base=r reads:
  LDS[r+0*32] = Q[r,0]  ✗ stride is 32, not 8!
```
This doesn't work directly.

### Option 2: Interleaved Rows (8-row blocks)
Store Q in 8-row interleaved blocks:
```
Q[r,k] at LDS[(r%8) + (r/8)*8*128 + k*8]

For rows 0-7 (block 0):
  Q[0,k] at LDS[0 + k*8]
  Q[1,k] at LDS[1 + k*8]
  ...
  Q[7,k] at LDS[7 + k*8]

TR8 for row 0, k=0..7:
  base = 0
  LDS[0+0*8=0], LDS[0+1*8=8], LDS[0+2*8=16], ... = Q[0,0], Q[0,1], Q[0,2], ...
  ✓ This works!
```

### Layout Formula:
```
LDS_offset(row, k) = (row % 8) + (row / 8) * (8 * K_DIM) + k * 8

For Q[32×128]:
  row_block = row / 8     (0-3 for 32 rows)
  row_in_block = row % 8  (0-7)
  LDS_offset = row_in_block + row_block * 1024 + k * 8
```

## Global-to-LDS Load Pattern

To create this layout from row-major global memory:

### Method 1: Direct Scatter (expensive)
Each thread computes its swizzled destination.
Problem: Irregular access pattern, poor memory coalescing.

### Method 2: Load Row-Major, Transpose in LDS
1. Load 16 bytes (16 FP8) per thread into VGPRs
2. Write to LDS with swizzled pattern
Problem: Extra LDS traffic.

### Method 3: Buffer_load with m0 Swizzle
Use buffer_load...lds with m0 offset to create swizzled layout.
Need to derive m0 pattern that produces interleaved-8 layout.

## What BF16 Does (for reference)

BF16 uses:
- Q reads: `ds_read_b64` (NOT TR16) with offsets 0,8,32,40,64,72,96,104
- The pattern: `(k/2)*32 + (k%2)*8` for k=0..7

This is a different swizzle pattern for 16-bit elements.
For FP8 (8-bit elements), the pattern scales differently.

## Conclusion

For TR8 to work, Q[32×128] must be stored as:
```
LDS_offset(row, k) = (row % 8) + (row / 8) * 1024 + k * 8
```

**The global load must scatter data into this interleaved pattern.**

This is fundamentally different from the BF16 m0 swizzle pattern
which is designed for TR16 (16-bit transpose reads).

## Next Steps
1. Either: Implement correct interleaved load pattern for TR8
2. Or: Use plain ds_read_b64 with BF16-style swizzle (like BF16 does for Q/K)

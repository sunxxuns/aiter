---
name: triton-fp8-tr8-ground-truth
description: Ground-truth, line-anchored excerpts from Triton-generated gfx950 FP8 MFMA+TR8 assembly (`triton_fp8_fmha.s`). Use as a canonical reference when porting LDS write swizzles, TR8 base/offsets, and MFMA operand roles into `fwd_fp8_scaffold.s`.
---

# Triton FP8 TR8 ground truth (gfx950)

## Source artifact (single source of truth)

- **File**: `aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s`
- **Provenance**: Triton-generated AMDGCN asm (contains `.loc` references to `bench_triton_fmha.py`).

Everything below is **verbatim evidence** from that file with **line numbers** so it can be audited quickly.

## Ground-truth facts (indexed)

### GT-01: TR8 loads used for FP8 MFMA A operand

`ds_read_b64_tr_b8` appears and is used to feed `v_mfma_f32_32x32x64_f8f6f4`.

- Evidence (TR8 reads + MFMA):

```679:724:aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s
ds_read_b64_tr_b8 v[74:75], v145
ds_read_b64_tr_b8 v[76:77], v145 offset:1088
ds_read_b64_tr_b8 v[78:79], v145 offset:4096
ds_read_b64_tr_b8 v[80:81], v145 offset:5184
// ...
v_mfma_f32_32x32x64_f8f6f4 v[50:65], v[74:81], v[66:73], v[50:65]
// ...
ds_read_b64_tr_b8 v[74:75], v153
ds_read_b64_tr_b8 v[76:77], v154
ds_read_b64_tr_b8 v[78:79], v155
ds_read_b64_tr_b8 v[80:81], v156
// ...
v_mfma_f32_32x32x64_f8f6f4 v[2:17], v[74:81], v[66:73], v[2:17]
```

### GT-02: TR8 offset sets (exact)

Triton uses **two distinct offset patterns** for the `v145` base, and **zero-offset reads** for the other bases.

- **GT-02.A (base = `v145`)**:
  - `{0, 1088, 4096, 5184}` (lines 679–682)
  - `{64, 1024, 4160, 5120}` (lines 710–713)

- **GT-02.B (bases = `v149..v152`)**: offsets all `0` (lines 697–700)

- **GT-02.C (bases = `v153..v156`)**: offsets all `0` (lines 718–721)

### GT-03: MFMA operand roles (Triton order)

For `v_mfma_f32_32x32x64_f8f6f4`, Triton uses:

- **A operand**: `v[74:81]` (the TR8-loaded fragment)
- **B operand**: `v[66:73]`
- **Accumulators**: `v[50:65]` (and similarly other accumulator ranges)

- Evidence:

```694:709:aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x64_f8f6f4 v[50:65], v[74:81], v[66:73], v[50:65]
// ...
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x64_f8f6f4 v[34:49], v[74:81], v[66:73], v[34:49]
```

### GT-04: LDS write pattern immediately preceding TR8 reads (observed)

Just before TR8 reads, Triton writes two 128-bit vectors into LDS at base `v142` with offsets `0` and `4096`, then barriers, then issues TR8 reads.

- Evidence:

```672:682:aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s
s_waitcnt vmcnt(1)
ds_write_b128 v142, v[98:101]
s_waitcnt vmcnt(0)
ds_write_b128 v142, v[102:105] offset:4096
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[74:75], v145
ds_read_b64_tr_b8 v[76:77], v145 offset:1088
ds_read_b64_tr_b8 v[78:79], v145 offset:4096
ds_read_b64_tr_b8 v[80:81], v145 offset:5184
```

### GT-05: One instance of LDS base swizzle for writes uses bitop3:0x78

Triton computes the LDS write base `v142` via `v_bitop3_b32 ... bitop3:0x78`.

- Evidence:

```107:123:aiter/hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s
s_movk_i32 s16, 0x70
v_bitop3_b32 v1, v28, v0, s16 bitop3:0x78
v_add_u32_e32 v142, 0, v1
// ...
ds_write_b128 v142, v[4:7]
ds_write_b128 v142, v[8:11] offset:4096
ds_write_b128 v142, v[12:15] offset:8192
ds_write_b128 v142, v[16:19] offset:12288
```

## How to audit quickly (copy/paste friendly)

- **Find all TR8 reads**:
  - search `ds_read_b64_tr_b8` in `triton_fp8_fmha.s` (16 matches)
- **Find the PV MFMA operand order**:
  - search `v_mfma_f32_32x32x64_f8f6f4 v[50:65], v[74:81], v[66:73]`
- **Find the LDS write swizzle**:
  - search `bitop3:0x78` near `v142`


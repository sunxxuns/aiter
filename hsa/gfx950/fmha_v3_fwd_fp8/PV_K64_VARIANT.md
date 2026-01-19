# PV K=64 Scaffold Variant (Experimental, Slower)

This captures the attempted PV K=64 path with Triton-style V LDS swizzle.
It regressed vs the current K=16 PV baseline but is saved for reference.

## Observed performance
- PV K=64 + swizzled V store/read: ~12.95 ms (eq ~1633 TF/s)
- Baseline K=16 PV: ~10.77 ms (eq ~1963 TF/s)

## How to re-apply the variant

### 1) LDS layout (pad V to 64 rows)
Update constants in `fwd_fp8_scaffold.s`:
```
.set K_LDS0, 33792            // 32×128 (row-major)
.set V_LDS0, 37888            // 32×128 swizzled, padded to 64 rows for K=64
.set K_LDS1, 46080            // ping-pong (after padded V_LDS0)
.set V_LDS1, 50176            // ping-pong (padded)
.set LDS_SIZE, 58368          // aligned
```
and metadata:
```
.amdhsa_group_segment_fixed_size 58368
...
.group_segment_fixed_size: 58368
```

### 2) Launch LDS size
In `test_scaffold.py`:
```
lds_bytes = 58368
```

### 3) Swizzled V LDS store address
Replace thread indexing block:
```
v_and_b32_e32 v62, 0xFF, v60         // tid_in_tile (0-255)
v_lshlrev_b32_e32 v61, 4, v62        // (tid & 255) * 16 (prefetch vaddr)
v_mov_b32_e32 v198, 0x70
v_bitop3_b32 v199, v61, v62, v198 bitop3:0x78
```

Use `v199` for V LDS buffer loads (both prefetch and loop prefetch):
```
v_mov_b32_e32 v13, v199
s_mov_b32 m0, V_LDS0
buffer_load_dwordx4 v13, s[16:19], s31 offen lds
```
and similarly for ping-pong prefetch.

### 4) PV K=64 swizzled read block
Replace the PV block with the following:
```
// PV MFMA using Triton-style swizzled V reads (K=64 MFMA)
// Compute swizzled base (v10 in Triton) from tid_in_tile (v62)
v_lshlrev_b32_e32 v200, 3, v62
v_and_b32_e32 v200, 8, v200
v_lshlrev_b32_e32 v201, 6, v62
v_and_b32_e32 v201, 0xb80, v201
v_and_b32_e32 v202, 16, v62
v_or_b32_e32 v201, v201, v202
v_bitop3_b32 v200, v201, v202, v200 bitop3:0x36

v_xor_b32_e32 v201, 32, v200
v_xor_b32_e32 v202, 0x460, v200
v_xor_b32_e32 v203, 0x1020, v200
v_xor_b32_e32 v204, 0x1460, v200
v_xor_b32_e32 v205, 0x60, v200
v_xor_b32_e32 v206, 0x420, v200
v_xor_b32_e32 v207, 0x1060, v200
v_xor_b32_e32 v208, 0x1420, v200

// Add V LDS base to swizzled offsets
v_add_u32_e32 v200, v56, v200
v_add_u32_e32 v201, v56, v201
v_add_u32_e32 v202, v56, v202
v_add_u32_e32 v203, v56, v203
v_add_u32_e32 v204, v56, v204
v_add_u32_e32 v205, v56, v205
v_add_u32_e32 v206, v56, v206
v_add_u32_e32 v207, v56, v207
v_add_u32_e32 v208, v56, v208

.macro READ_V_K64_OFF base, off0, off1, off2, off3
    ds_read_b64_tr_b8 v[0:1], \base offset:\off0
    ds_read_b64_tr_b8 v[2:3], \base offset:\off1
    ds_read_b64_tr_b8 v[4:5], \base offset:\off2
    ds_read_b64_tr_b8 v[6:7], \base offset:\off3
.endm

.macro READ_V_K64_BASES b0, b1, b2, b3
    ds_read_b64_tr_b8 v[0:1], \b0
    ds_read_b64_tr_b8 v[2:3], \b1
    ds_read_b64_tr_b8 v[4:5], \b2
    ds_read_b64_tr_b8 v[6:7], \b3
.endm

// Convert P (v[32:47]) to FP8 packed (v48-v55), zero-extend to K=64
v_mov_b32_e32 v59, 0x05040100
v_cvt_pk_fp8_f32 v48, v32, v33
v_cvt_pk_fp8_f32 v49, v34, v35
v_perm_b32 v48, v48, v49, v59

v_cvt_pk_fp8_f32 v49, v36, v37
v_cvt_pk_fp8_f32 v50, v38, v39
v_perm_b32 v49, v49, v50, v59

v_cvt_pk_fp8_f32 v50, v40, v41
v_cvt_pk_fp8_f32 v51, v42, v43
v_perm_b32 v50, v50, v51, v59

v_cvt_pk_fp8_f32 v51, v44, v45
v_cvt_pk_fp8_f32 v52, v46, v47
v_perm_b32 v51, v51, v52, v59

v_mov_b32_e32 v52, 0
v_mov_b32_e32 v53, 0
v_mov_b32_e32 v54, 0
v_mov_b32_e32 v55, 0

// PV MFMA (K=64), 4 column blocks
READ_V_K64_OFF v200, 0, 1088, 4096, 5184
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x64_f8f6f4 v[64:79], v[0:7], v[48:55], v[64:79]

READ_V_K64_BASES v201, v202, v203, v204
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x64_f8f6f4 v[80:95], v[0:7], v[48:55], v[80:95]

READ_V_K64_OFF v200, 64, 1024, 4160, 5120
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x64_f8f6f4 v[96:111], v[0:7], v[48:55], v[96:111]

READ_V_K64_BASES v205, v206, v207, v208
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x64_f8f6f4 v[112:127], v[0:7], v[48:55], v[112:127]
```

### 5) Run with these env vars
```
SCAFFOLD_PV_K=64 SCAFFOLD_PV_MFMA=4
```

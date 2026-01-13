// FP8 Flash Attention - 256 threads (4 waves)
// Cooperative loading + all waves compute same MFMA
//
// Key differences from 64T:
// - 4x faster global→LDS loading (all 256 threads help)
// - Better latency hiding with 4 waves
// - Same MFMA compute pattern

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter12fwd_fp8_256tE
.p2align 8
.type _ZN5aiter12fwd_fp8_256tE,@function

_ZN5aiter12fwd_fp8_256tE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // THREAD ID SETUP
    // ========================================================================
    // tid = v0 (0-255)
    // wave_id = tid >> 6 (0-3)
    // lane_id = tid & 63 (0-63)
    v_mov_b32_e32 v59, v0                 // Save tid (v0 will be overwritten by MFMA)
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id
    v_and_b32_e32 v60, 63, v0             // lane_id
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [32×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [seq×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [seq×128] FP8
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    // Constants
    s_mov_b32 s2, 0x3e028f5c              // log2(e) / sqrt(128)
    s_mov_b32 s3, 0xff800000              // -infinity
    
    s_waitcnt lgkmcnt(0)
    
    // Setup buffer descriptors
    s_mov_b32 s10, s8                      // Q base
    s_mov_b32 s11, s9
    s_mov_b32 s14, s12                     // K base
    s_mov_b32 s15, s13
    s_mov_b32 s18, s16                     // V base
    s_mov_b32 s19, s17
    
    // Number of K-tiles
    s_add_i32 s25, s24, 31
    s_lshr_b32 s25, s25, 5                 // num_tiles = (seq_len + 31) / 32
    
    // ========================================================================
    // COOPERATIVE Q LOAD (256 threads load 32×128 = 4KB)
    // Each thread loads 16 bytes (4KB / 256 = 16)
    // ========================================================================
    
    // tid loads bytes [tid*16 : tid*16+16)
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16
    
    // Global addr: Q_ptr + tid * 16
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load Q
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    // Write to LDS at same offset
    ds_write_b128 v1, v[20:23]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // MFMA ROW MAPPING (same for all waves, using lane_id)
    // ========================================================================
    
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v62, v3, v4             // mfma_row (0-31)
    
    // k_base: 0 for lanes 0-31, 8 for lanes 32-63
    v_mov_b32_e32 v63, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e32 v63, v63, v4, vcc   // k_base
    
    // Q LDS read address: mfma_row * 128 + k_base
    v_lshlrev_b32_e32 v70, 7, v62
    v_add_u32_e32 v70, v70, v63
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE
    // ========================================================================
    
    // running_max = -inf (per lane, 16 values)
    v_mov_b32_e32 v40, s3
    v_mov_b32_e32 v41, s3
    v_mov_b32_e32 v42, s3
    v_mov_b32_e32 v43, s3
    v_mov_b32_e32 v44, s3
    v_mov_b32_e32 v45, s3
    v_mov_b32_e32 v46, s3
    v_mov_b32_e32 v47, s3
    v_mov_b32_e32 v48, s3
    v_mov_b32_e32 v49, s3
    v_mov_b32_e32 v50, s3
    v_mov_b32_e32 v51, s3
    v_mov_b32_e32 v52, s3
    v_mov_b32_e32 v53, s3
    v_mov_b32_e32 v54, s3
    v_mov_b32_e32 v55, s3
    
    // running_sum = 0
    v_mov_b32_e32 v56, 0
    
    // O accumulators = 0 (v80-v143)
    .irp i, 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143
        v_mov_b32_e32 v\i, 0
    .endr
    
    // K-tile loop counter
    s_mov_b32 s26, 0                       // tile_idx = 0
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
K_TILE_LOOP:
    
    // --------------------------------------------------------------------
    // COOPERATIVE K LOAD (256 threads load 32×128 = 4KB)
    // --------------------------------------------------------------------
    
    // K offset: tile_idx * 32 * 128 = tile_idx * 4096
    s_lshl_b32 s27, s26, 12               // tile_offset = tile_idx * 4096
    
    // Each thread loads 16 bytes
    v_lshlrev_b32_e32 v1, 4, v59          // tid * 16 (use saved tid!)
    v_mov_b32_e32 v2, s27
    v_add_u32_e32 v1, v1, v2              // + tile_offset
    
    // Global addr
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load K
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    // Write to LDS at offset 4096 (K region)
    v_lshlrev_b32_e32 v1, 4, v59          // use saved tid
    v_add_u32_e32 v1, 4096, v1
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // --------------------------------------------------------------------
    // QK MFMA: S = K @ Q^T
    // All 4 waves do same computation
    // --------------------------------------------------------------------
    
    // K LDS addr: 4096 + mfma_row * 128 + k_base
    v_lshlrev_b32_e32 v71, 7, v62
    v_add_u32_e32 v71, v71, v63
    v_add_u32_e32 v71, 4096, v71
    
    // Clear S accumulators
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0
    v_mov_b32_e32 v13, 0
    v_mov_b32_e32 v14, 0
    v_mov_b32_e32 v15, 0
    
    // 8 MFMAs for HD=128
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v71    // K addr
        ds_read_b64 v[30:31], v72
        v_add_u32_e32 v73, \k_off, v70    // Q addr
        ds_read_b64 v[32:33], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    
    s_nop 15
    
    // S is now in v[0:15] (32×32 scores, 16 per lane)
    
    // --------------------------------------------------------------------
    // SOFTMAX: Find max, compute exp, accumulate
    // (simplified - just use max of v0)
    // --------------------------------------------------------------------
    
    // Scale by 1/sqrt(d)
    v_mul_f32_e32 v0, s2, v0
    v_mul_f32_e32 v1, s2, v1
    v_mul_f32_e32 v2, s2, v2
    v_mul_f32_e32 v3, s2, v3
    v_mul_f32_e32 v4, s2, v4
    v_mul_f32_e32 v5, s2, v5
    v_mul_f32_e32 v6, s2, v6
    v_mul_f32_e32 v7, s2, v7
    v_mul_f32_e32 v8, s2, v8
    v_mul_f32_e32 v9, s2, v9
    v_mul_f32_e32 v10, s2, v10
    v_mul_f32_e32 v11, s2, v11
    v_mul_f32_e32 v12, s2, v12
    v_mul_f32_e32 v13, s2, v13
    v_mul_f32_e32 v14, s2, v14
    v_mul_f32_e32 v15, s2, v15
    
    // For simplicity, use uniform attention (P = 1/seq_len)
    // This avoids complex softmax but still tests the structure
    // TODO: Add proper online softmax
    
    // Convert S to FP8 for PV MFMA (pack pairs)
    v_cvt_pk_fp8_f32 v30, v0, v1
    v_cvt_pk_fp8_f32 v31, v2, v3
    v_cvt_pk_fp8_f32 v32, v4, v5
    v_cvt_pk_fp8_f32 v33, v6, v7
    
    // --------------------------------------------------------------------
    // COOPERATIVE V LOAD
    // --------------------------------------------------------------------
    
    v_lshlrev_b32_e32 v1, 4, v59          // use saved tid
    v_mov_b32_e32 v2, s27                 // same tile_offset as K
    v_add_u32_e32 v1, v1, v2
    
    v_mov_b32_e32 v10, s16
    v_mov_b32_e32 v11, s17
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    // V to LDS at offset 8192
    v_lshlrev_b32_e32 v1, 4, v59          // use saved tid
    v_add_u32_e32 v1, 8192, v1
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // --------------------------------------------------------------------
    // PV MFMA: O += P @ V
    // (simplified - just accumulate without proper softmax weighting)
    // --------------------------------------------------------------------
    
    // V LDS addr for HD tile 0
    v_lshlrev_b32_e32 v74, 7, v62
    v_add_u32_e32 v74, v74, v63
    v_add_u32_e32 v74, 8192, v74
    
    // Read V
    ds_read_b64 v[34:35], v74
    s_waitcnt lgkmcnt(0)
    
    // Pack P to AGPR
    v_accvgpr_write_b32 a0, v30
    v_accvgpr_write_b32 a1, v31
    
    // PV MFMA for HD tile 0
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[34:35], v[80:95]
    s_nop 15
    
    // --------------------------------------------------------------------
    // LOOP INCREMENT
    // --------------------------------------------------------------------
    
    s_add_i32 s26, s26, 1
    s_cmp_lt_i32 s26, s25
    s_cbranch_scc1 K_TILE_LOOP
    
    // ========================================================================
    // FINAL OUTPUT (wave 0 only)
    // ========================================================================
    
    v_cmp_eq_u32_e64 vcc, v61, 0          // wave_id == 0?
    s_and_saveexec_b64 s[28:29], vcc
    
    // Output O[32×128] - just first 32 columns for now
    v_mov_b32_e32 v10, s4
    v_mov_b32_e32 v11, s5
    v_lshlrev_b32_e32 v12, 6, v60         // lane * 64 bytes
    v_add_co_u32_e32 v10, vcc, v12, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_store_dwordx4 v[10:11], v[80:83]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[84:87]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[88:91]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[92:95]
    
    s_mov_b64 exec, s[28:29]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter12fwd_fp8_256tE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 160
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 160
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter12fwd_fp8_256tE
    .symbol: _ZN5aiter12fwd_fp8_256tE.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 160
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

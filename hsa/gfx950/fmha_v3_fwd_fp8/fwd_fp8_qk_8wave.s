// FP8 QK with 8 waves (512 threads)
// Each pair of waves processes different Q-rows, sharing K data
// Double the parallelism vs 4-wave version

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136
// Q: 64 rows x 136 pitch = 8704 bytes (for 8 waves processing 64 Q-rows)
.set Q_LDS, 0
.set K_LDS_A, 8704
.set K_LDS_B, 13056                 // 8704 + 4352
.set LDS_SIZE, 20480                // ~20KB needed

.text
.globl _ZN5aiter15fwd_fp8_qk_8waveE
.p2align 8
.type _ZN5aiter15fwd_fp8_qk_8waveE,@function

_ZN5aiter15fwd_fp8_qk_8waveE:
    s_mov_b64 exec, -1
    
    s_load_dwordx2 s[4:5], s[0:1], 0
    s_load_dwordx2 s[8:9], s[0:1], 8
    s_load_dwordx2 s[12:13], s[0:1], 16
    s_load_dword s20, s[0:1], 24
    s_load_dword s21, s[0:1], 28
    s_waitcnt lgkmcnt(0)
    
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    v_and_b32_e32 v60, 63, v0          // lane within wave (0-63)
    v_lshrrev_b32_e32 v61, 6, v0       // wave_id (0-7)
    v_and_b32_e32 v62, 511, v0         // thread_id (0-511)
    
    // ========================================================================
    // Global load offsets for Q (512 threads load 64 rows x 128 cols)
    // Each thread loads 16 bytes (128 FP8 elements total = 8192 bytes)
    // row = thread_id / 8, col = thread_id % 8
    // ========================================================================
    v_lshrrev_b32_e32 v1, 3, v62       // row = thread_id / 8 (0-63)
    v_and_b32_e32 v2, 7, v62           // col = thread_id % 8 (0-7)
    v_lshlrev_b32_e32 v3, 7, v1        // row * 128 (global stride)
    v_lshlrev_b32_e32 v4, 4, v2        // col * 16
    v_add_u32_e32 v50, v3, v4          // global offset within tile
    
    // LDS offset with pitch for Q (64 rows)
    v_mov_b32_e32 v5, PITCH
    v_mul_lo_u32 v51, v1, v5           // row * PITCH
    v_add_u32_e32 v51, v51, v4         // + col * 16
    
    // ========================================================================
    // Load Q (64 rows) - each of 512 threads loads 16 bytes
    // Only threads 0-511 load (all threads in 8 waves)
    // ========================================================================
    s_lshl_b32 s22, s21, 13            // q_tile_idx * 8192 (64 rows * 128 bytes)
    v_add_u32_e32 v52, s22, v50
    buffer_load_dwordx4 v[80:83], v52, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v53, Q_LDS, v51
    ds_write_b128 v53, v[80:83]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // MFMA read addresses - each wave processes different 32-row section
    // waves 0-3 -> Q rows 0-31, waves 4-7 -> Q rows 32-63
    // ========================================================================
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v63, v3, v4          // row within 32-row MFMA tile
    
    // Add 32 for waves 4-7
    v_cmp_ge_u32_e64 s[26:27], v61, 4
    v_cndmask_b32_e64 v65, 0, 32, s[26:27]
    v_add_u32_e32 v63, v63, v65        // row index 0-63
    
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc   // k_offset
    
    v_mov_b32_e32 v1, PITCH
    v_mul_lo_u32 v70, v63, v1
    v_add_u32_e32 v70, v70, v64
    v_add_u32_e32 v70, Q_LDS, v70      // Q read address
    
    // K base (K_LDS_A - Q_LDS relative offset)
    // For K, all waves read from same K data (32 rows)
    // K read row is same as v63 mod 32
    v_and_b32_e32 v66, 31, v63         // K row = Q row mod 32
    v_mul_lo_u32 v71, v66, v1
    v_add_u32_e32 v71, v71, v64
    v_add_u32_e32 v71, K_LDS_A, v71    // K read address
    
    // num_k_tiles
    s_add_u32 s23, s20, 31
    s_lshr_b32 s23, s23, 5
    
    // Clear accumulators
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // Pre-load Q chunks (reused across all K tiles)
    ds_read_b64 v[16:17], v70 offset:0
    ds_read_b64 v[18:19], v70 offset:16
    ds_read_b64 v[20:21], v70 offset:32
    ds_read_b64 v[22:23], v70 offset:48
    ds_read_b64 v[24:25], v70 offset:64
    ds_read_b64 v[26:27], v70 offset:80
    ds_read_b64 v[28:29], v70 offset:96
    ds_read_b64 v[30:31], v70 offset:112
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // K-LOOP: Load K tile, chain MFMAs
    // K is loaded by first 256 threads only (32 rows x 128 cols = 4096 bytes)
    // ========================================================================
    
    // K load offset (only threads 0-255 participate)
    s_mov_b32 s30, 256
    v_cmp_lt_u32_e64 s[28:29], v62, s30
    v_lshrrev_b32_e32 v73, 3, v62      // row = thread_id / 8 (0-31 for K)
    v_and_b32_e32 v74, 7, v62          // col = thread_id % 8
    v_lshlrev_b32_e32 v75, 7, v73      // row * 128
    v_lshlrev_b32_e32 v76, 4, v74      // col * 16
    v_add_u32_e32 v77, v75, v76        // K global offset within tile
    
    // K LDS offset
    v_mul_lo_u32 v78, v73, v5          // row * PITCH
    v_add_u32_e32 v78, v78, v76        // + col * 16
    
    // Load K tile 0
    s_mov_b32 s24, 0
    buffer_load_dwordx4 v[84:87], v77, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    
    // Only threads 0-255 write K to LDS
    s_mov_b64 exec, s[28:29]
    v_add_u32_e32 v79, K_LDS_A, v78
    ds_write_b128 v79, v[84:87]
    s_mov_b64 exec, -1
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_cmp_eq_u32 s23, 1
    s_cbranch_scc1 LAST_TILE
    
    s_mov_b32 s24, 1

PIPELINE_LOOP:
    // Start async load of next K tile
    s_lshl_b32 s25, s24, 12
    v_add_u32_e32 v80, s25, v77
    buffer_load_dwordx4 v[84:87], v80, s[12:15], 0 offen
    
    // Pre-load K chunks from buffer A
    ds_read_b64 v[32:33], v71 offset:0
    ds_read_b64 v[34:35], v71 offset:16
    ds_read_b64 v[36:37], v71 offset:32
    ds_read_b64 v[38:39], v71 offset:48
    ds_read_b64 v[40:41], v71 offset:64
    ds_read_b64 v[42:43], v71 offset:80
    ds_read_b64 v[44:45], v71 offset:96
    ds_read_b64 v[46:47], v71 offset:112
    s_waitcnt lgkmcnt(0)
    
    // Chain 8 MFMAs
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    
    // Wait for K load, write to buffer B
    s_waitcnt vmcnt(0)
    s_mov_b64 exec, s[28:29]
    v_add_u32_e32 v79, K_LDS_B, v78
    ds_write_b128 v79, v[84:87]
    s_mov_b64 exec, -1
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s24, s24, 1
    s_cmp_ge_u32 s24, s23
    s_cbranch_scc1 LAST_TILE_B
    
    // Start async load of next K tile
    s_lshl_b32 s25, s24, 12
    v_add_u32_e32 v80, s25, v77
    buffer_load_dwordx4 v[84:87], v80, s[12:15], 0 offen
    
    // K read from buffer B
    v_add_u32_e32 v72, K_LDS_B - K_LDS_A, v71
    
    // Pre-load K chunks
    ds_read_b64 v[32:33], v72 offset:0
    ds_read_b64 v[34:35], v72 offset:16
    ds_read_b64 v[36:37], v72 offset:32
    ds_read_b64 v[38:39], v72 offset:48
    ds_read_b64 v[40:41], v72 offset:64
    ds_read_b64 v[42:43], v72 offset:80
    ds_read_b64 v[44:45], v72 offset:96
    ds_read_b64 v[46:47], v72 offset:112
    s_waitcnt lgkmcnt(0)
    
    // Chain 8 MFMAs
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    
    // Wait for K load, write to buffer A
    s_waitcnt vmcnt(0)
    s_mov_b64 exec, s[28:29]
    v_add_u32_e32 v79, K_LDS_A, v78
    ds_write_b128 v79, v[84:87]
    s_mov_b64 exec, -1
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s24, s24, 1
    s_cmp_lt_u32 s24, s23
    s_cbranch_scc1 PIPELINE_LOOP
    
LAST_TILE:
    ds_read_b64 v[32:33], v71 offset:0
    ds_read_b64 v[34:35], v71 offset:16
    ds_read_b64 v[36:37], v71 offset:32
    ds_read_b64 v[38:39], v71 offset:48
    ds_read_b64 v[40:41], v71 offset:64
    ds_read_b64 v[42:43], v71 offset:80
    ds_read_b64 v[44:45], v71 offset:96
    ds_read_b64 v[46:47], v71 offset:112
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    s_branch STORE_OUTPUT

LAST_TILE_B:
    v_add_u32_e32 v72, K_LDS_B - K_LDS_A, v71
    
    ds_read_b64 v[32:33], v72 offset:0
    ds_read_b64 v[34:35], v72 offset:16
    ds_read_b64 v[36:37], v72 offset:32
    ds_read_b64 v[38:39], v72 offset:48
    ds_read_b64 v[40:41], v72 offset:64
    ds_read_b64 v[42:43], v72 offset:80
    ds_read_b64 v[44:45], v72 offset:96
    ds_read_b64 v[46:47], v72 offset:112
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    
STORE_OUTPUT:
    // Output: each thread stores 64 bytes at different offset
    // waves 0-3 store to output rows 0-31, waves 4-7 to rows 32-63
    v_lshlrev_b32_e32 v48, 6, v60      // lane * 64
    v_lshlrev_b32_e32 v49, 11, v65     // wave_group * 2048 (32 rows * 64 bytes)
    v_add_u32_e32 v48, v48, v49
    
    buffer_store_dwordx4 v[0:3], v48, s[4:7], 0 offen
    v_add_u32_e32 v49, 16, v48
    buffer_store_dwordx4 v[4:7], v49, s[4:7], 0 offen
    v_add_u32_e32 v49, 32, v48
    buffer_store_dwordx4 v[8:11], v49, s[4:7], 0 offen
    v_add_u32_e32 v49, 48, v48
    buffer_store_dwordx4 v[12:15], v49, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter15fwd_fp8_qk_8waveE
    .amdhsa_group_segment_fixed_size 20480
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 96
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 96
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter15fwd_fp8_qk_8waveE
    .symbol: _ZN5aiter15fwd_fp8_qk_8waveE.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 20480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 96
    .max_flat_workgroup_size: 512
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 24, .value_kind: by_value}
      - {.name: q_tile_idx, .size: 4, .offset: 28, .value_kind: by_value}
...
.end_amdgpu_metadata

// FP8 QK with 4 Q-tiles per block (128 Q rows)
// FIXED: Each wave computes 1 Q-tile (not all 4 redundantly)
// 256 threads (4 waves), pitch-136 layout

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136
.set Q_LDS_0, 0
.set Q_LDS_1, 4352
.set Q_LDS_2, 8704
.set Q_LDS_3, 13056
.set K_LDS_A, 17408
.set K_LDS_B, 21760
.set LDS_SIZE, 28672

.text
.globl _ZN5aiter19fwd_fp8_qk_4qtile_v2E
.p2align 8
.type _ZN5aiter19fwd_fp8_qk_4qtile_v2E,@function

_ZN5aiter19fwd_fp8_qk_4qtile_v2E:
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
    
    v_and_b32_e32 v200, 63, v0           // wave_tid = tid % 64
    v_lshrrev_b32_e32 v201, 6, v0        // wave_id = tid / 64 (0-3)
    v_and_b32_e32 v202, 255, v0          // block_tid = tid % 256
    
    v_mov_b32_e32 v5, PITCH
    
    // ========================================================================
    // Load 4 Q tiles (128 rows total) - all threads participate
    // ========================================================================
    v_lshrrev_b32_e32 v1, 3, v202
    v_and_b32_e32 v2, 7, v202
    v_lshlrev_b32_e32 v3, 7, v1
    v_lshlrev_b32_e32 v4, 4, v2
    v_add_u32_e32 v203, v3, v4          // global offset base
    
    v_mul_lo_u32 v204, v1, v5
    v_add_u32_e32 v204, v204, v4        // LDS offset base
    
    // Q base = q_tile_idx * 16384 (128 rows × 128 bytes)
    s_lshl_b32 s22, s21, 14
    
    // Load Q tile 0
    v_add_u32_e32 v6, s22, v203
    buffer_load_dwordx4 v[180:183], v6, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v7, Q_LDS_0, v204
    ds_write_b128 v7, v[180:183]
    
    // Load Q tile 1
    v_add_u32_e32 v6, 4096, v6
    buffer_load_dwordx4 v[180:183], v6, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v7, Q_LDS_1, v204
    ds_write_b128 v7, v[180:183]
    
    // Load Q tile 2
    v_add_u32_e32 v6, 4096, v6
    buffer_load_dwordx4 v[180:183], v6, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v7, Q_LDS_2, v204
    ds_write_b128 v7, v[180:183]
    
    // Load Q tile 3
    v_add_u32_e32 v6, 4096, v6
    buffer_load_dwordx4 v[180:183], v6, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v7, Q_LDS_3, v204
    ds_write_b128 v7, v[180:183]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // MFMA addresses - now wave-dependent!
    // ========================================================================
    v_and_b32_e32 v1, 3, v200
    v_lshrrev_b32_e32 v2, 3, v200
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v200
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v205, v3, v4          // MFMA row (0-31)
    
    v_cmp_ge_u32_e64 vcc, v200, 32
    v_cndmask_b32_e64 v206, 0, 8, vcc   // k_offset
    
    v_mul_lo_u32 v207, v205, v5
    v_add_u32_e32 v207, v207, v206      // base LDS read offset
    
    // Each wave selects its Q-tile based on wave_id
    // wave_id * Q_TILE_SIZE = wave_id * 4352
    v_lshlrev_b32_e32 v208, 12, v201    // wave_id * 4096 (approx)
    v_lshlrev_b32_e32 v209, 8, v201     // wave_id * 256
    v_add_u32_e32 v208, v208, v209      // wave_id * 4352
    v_add_u32_e32 v210, v207, v208      // Q LDS address for this wave's tile
    
    s_add_u32 s23, s20, 31
    s_lshr_b32 s23, s23, 5
    
    // Clear output (16 VGPRs per wave, only 1 tile now)
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // Pre-load this wave's Q tile into VGPRs v[16:31]
    ds_read_b64 v[16:17], v210 offset:0
    ds_read_b64 v[18:19], v210 offset:16
    ds_read_b64 v[20:21], v210 offset:32
    ds_read_b64 v[22:23], v210 offset:48
    ds_read_b64 v[24:25], v210 offset:64
    ds_read_b64 v[26:27], v210 offset:80
    ds_read_b64 v[28:29], v210 offset:96
    ds_read_b64 v[30:31], v210 offset:112
    s_waitcnt lgkmcnt(0)
    
    // K addresses (shared across all waves)
    v_add_u32_e32 v214, K_LDS_A, v207
    
    // K load setup
    v_lshrrev_b32_e32 v1, 3, v202
    v_and_b32_e32 v2, 7, v202
    v_lshlrev_b32_e32 v3, 7, v1
    v_lshlrev_b32_e32 v4, 4, v2
    v_add_u32_e32 v215, v3, v4
    v_mul_lo_u32 v216, v1, v5
    v_add_u32_e32 v216, v216, v4
    
    // Load K tile 0
    s_mov_b32 s24, 0
    buffer_load_dwordx4 v[180:183], v215, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v217, K_LDS_A, v216
    ds_write_b128 v217, v[180:183]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_cmp_eq_u32 s23, 1
    s_cbranch_scc1 LAST_TILE
    
    s_mov_b32 s24, 1

PIPELINE_LOOP:
    // Load next K tile (async)
    s_lshl_b32 s25, s24, 12
    v_add_u32_e32 v218, s25, v215
    buffer_load_dwordx4 v[180:183], v218, s[12:15], 0 offen
    
    // Load K from LDS into v[32:47]
    ds_read_b64 v[32:33], v214 offset:0
    ds_read_b64 v[34:35], v214 offset:16
    ds_read_b64 v[36:37], v214 offset:32
    ds_read_b64 v[38:39], v214 offset:48
    ds_read_b64 v[40:41], v214 offset:64
    ds_read_b64 v[42:43], v214 offset:80
    ds_read_b64 v[44:45], v214 offset:96
    ds_read_b64 v[46:47], v214 offset:112
    s_waitcnt lgkmcnt(0)
    
    // 8 MFMAs for this wave's Q-tile × K
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    
    // Write K to buffer B
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v217, K_LDS_B, v216
    ds_write_b128 v217, v[180:183]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s24, s24, 1
    s_cmp_ge_u32 s24, s23
    s_cbranch_scc1 LAST_TILE_B
    
    // Load next K tile (async)
    s_lshl_b32 s25, s24, 12
    v_add_u32_e32 v218, s25, v215
    buffer_load_dwordx4 v[180:183], v218, s[12:15], 0 offen
    
    // K from buffer B
    v_add_u32_e32 v219, K_LDS_B - K_LDS_A, v214
    
    ds_read_b64 v[32:33], v219 offset:0
    ds_read_b64 v[34:35], v219 offset:16
    ds_read_b64 v[36:37], v219 offset:32
    ds_read_b64 v[38:39], v219 offset:48
    ds_read_b64 v[40:41], v219 offset:64
    ds_read_b64 v[42:43], v219 offset:80
    ds_read_b64 v[44:45], v219 offset:96
    ds_read_b64 v[46:47], v219 offset:112
    s_waitcnt lgkmcnt(0)
    
    // 8 MFMAs
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    
    // Write K to buffer A
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v217, K_LDS_A, v216
    ds_write_b128 v217, v[180:183]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s24, s24, 1
    s_cmp_lt_u32 s24, s23
    s_cbranch_scc1 PIPELINE_LOOP
    
LAST_TILE:
    ds_read_b64 v[32:33], v214 offset:0
    ds_read_b64 v[34:35], v214 offset:16
    ds_read_b64 v[36:37], v214 offset:32
    ds_read_b64 v[38:39], v214 offset:48
    ds_read_b64 v[40:41], v214 offset:64
    ds_read_b64 v[42:43], v214 offset:80
    ds_read_b64 v[44:45], v214 offset:96
    ds_read_b64 v[46:47], v214 offset:112
    s_waitcnt lgkmcnt(0)
    s_branch DO_LAST_MFMA

LAST_TILE_B:
    v_add_u32_e32 v219, K_LDS_B - K_LDS_A, v214
    ds_read_b64 v[32:33], v219 offset:0
    ds_read_b64 v[34:35], v219 offset:16
    ds_read_b64 v[36:37], v219 offset:32
    ds_read_b64 v[38:39], v219 offset:48
    ds_read_b64 v[40:41], v219 offset:64
    ds_read_b64 v[42:43], v219 offset:80
    ds_read_b64 v[44:45], v219 offset:96
    ds_read_b64 v[46:47], v219 offset:112
    s_waitcnt lgkmcnt(0)

DO_LAST_MFMA:
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[16:17], v[32:33], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[18:19], v[34:35], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[20:21], v[36:37], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[22:23], v[38:39], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[24:25], v[40:41], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[26:27], v[42:43], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[28:29], v[44:45], v[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[46:47], v[0:15]
    
STORE_OUTPUT:
    // Output offset = wave_id * 4096 + wave_tid * 64
    v_lshlrev_b32_e32 v48, 12, v201     // wave_id * 4096
    v_lshlrev_b32_e32 v49, 6, v200      // wave_tid * 64
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
.amdhsa_kernel _ZN5aiter19fwd_fp8_qk_4qtile_v2E
    .amdhsa_group_segment_fixed_size 28672
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 220
    .amdhsa_next_free_sgpr 34
    .amdhsa_accum_offset 220
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter19fwd_fp8_qk_4qtile_v2E
    .symbol: _ZN5aiter19fwd_fp8_qk_4qtile_v2E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 28672
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 34
    .vgpr_count: 220
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 24, .value_kind: by_value}
      - {.name: q_tile_idx, .size: 4, .offset: 28, .value_kind: by_value}
...
.end_amdgpu_metadata

// FP8 QK MFMA with tiling and buffer_load...lds
// 256 threads (4 waves), K-loop for full seq_len
// Outputs S = Q @ K^T (F32 accumulator)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136                     // Row pitch for bank-conflict-free access
.set Q_LDS, 0                       // Q at LDS[0], 32*136=4352 bytes
.set K_LDS, 4352                    // K ping buffer
.set K_LDS2, 8704                   // K pong buffer (double buffering)
.set LDS_SIZE, 16384

.text
.globl _ZN5aiter15fwd_fp8_qk_tiledE
.p2align 8
.type _ZN5aiter15fwd_fp8_qk_tiledE,@function

_ZN5aiter15fwd_fp8_qk_tiledE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // ARGS: O_ptr, Q_ptr, K_ptr, seq_len, q_tile_idx
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr
    s_load_dword s20, s[0:1], 24          // seq_len
    s_load_dword s21, s[0:1], 28          // q_tile_idx (which 32-row Q tile)
    s_waitcnt lgkmcnt(0)
    
    // Buffer descriptors
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // Thread/wave IDs
    v_and_b32_e32 v60, 63, v0             // lane_id
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id (0-3)
    v_and_b32_e32 v62, 255, v0            // tid (0-255)
    
    // ========================================================================
    // COMPUTE LOAD OFFSETS
    // ========================================================================
    // Each thread loads 16 bytes (dwordx4)
    // 256 threads × 16 bytes = 4096 bytes = 32 rows × 128 cols
    // row = tid / 8, col_chunk = tid % 8
    
    v_lshrrev_b32_e32 v1, 3, v62          // row = tid / 8
    v_and_b32_e32 v2, 7, v62              // col_chunk = tid % 8
    
    // Global offset: row * 128 + col_chunk * 16
    v_lshlrev_b32_e32 v3, 7, v1           // row * 128
    v_lshlrev_b32_e32 v4, 4, v2           // col_chunk * 16
    v_add_u32_e32 v50, v3, v4             // v50 = global load offset within tile
    
    // LDS offset with pitch-136: row * 136 + col_chunk * 16
    v_mov_b32_e32 v5, PITCH
    v_mul_lo_u32 v51, v1, v5              // row * 136
    v_add_u32_e32 v51, v51, v4            // + col_chunk * 16
    // v51 = LDS offset (relative to tile base)
    
    // ========================================================================
    // LOAD Q TILE TO LDS (once, using buffer_load...lds)
    // ========================================================================
    
    // Q global offset for this tile: q_tile_idx * 32 * 128
    s_lshl_b32 s22, s21, 12               // q_tile_idx * 4096
    v_add_u32_e32 v52, s22, v50           // + thread offset
    
    // Set m0 for LDS destination
    v_add_u32_e32 v53, Q_LDS, v51
    v_readfirstlane_b32 s30, v53          // m0 needs scalar, use lane 0's value
    
    // Actually buffer_load...lds needs m0 to be same for all threads
    // So we need to load row by row with m0 stepping
    // Simplified: use flat approach first, optimize later
    
    buffer_load_dwordx4 v[20:23], v52, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v53, Q_LDS, v51
    ds_write_b128 v53, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE MFMA READ ADDRESSES
    // ========================================================================
    
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v63, v3, v4             // mfma_row
    
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc      // k_half
    
    // Q read base
    v_mov_b32_e32 v1, PITCH
    v_mul_lo_u32 v70, v63, v1
    v_add_u32_e32 v70, v70, v64
    v_add_u32_e32 v70, Q_LDS, v70         // v70 = Q read base
    
    // ========================================================================
    // K-LOOP: ITERATE OVER K TILES
    // ========================================================================
    
    // num_k_tiles = ceil(seq_len / 32)
    s_add_u32 s23, s20, 31
    s_lshr_b32 s23, s23, 5                // s23 = num_k_tiles
    
    // Clear S accumulators
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    s_mov_b32 s24, 0                      // k_tile_idx
    
K_LOOP:
    // Load K tile to LDS
    s_lshl_b32 s25, s24, 12               // k_tile_idx * 4096
    v_add_u32_e32 v54, s25, v50           // + thread offset
    
    buffer_load_dwordx4 v[24:27], v54, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v55, K_LDS, v51
    ds_write_b128 v55, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // K read base: same offset as Q but in K_LDS region
    // v70 = Q_LDS + mfma_row * PITCH + k_half
    // v71 = K_LDS + mfma_row * PITCH + k_half
    v_add_u32_e32 v71, K_LDS - Q_LDS, v70 // v71 = K read base
    
    // QK MFMA (8 iterations for HD=128)
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v70
        ds_read_b64 v[30:31], v72
        
        v_add_u32_e32 v73, \k_off, v71
        ds_read_b64 v[32:33], v73
        
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    s_nop 7
    
    s_barrier                              // Ensure all waves done before next K tile
    
    // Loop control
    s_add_u32 s24, s24, 1
    s_cmp_lt_u32 s24, s23
    s_cbranch_scc1 K_LOOP
    
    // ========================================================================
    // STORE S OUTPUT (F32, 16 values per lane)
    // ========================================================================
    
    v_lshlrev_b32_e32 v40, 6, v60         // lane * 64 bytes
    
    buffer_store_dwordx4 v[0:3], v40, s[4:7], 0 offen
    v_add_u32_e32 v42, 16, v40
    buffer_store_dwordx4 v[4:7], v42, s[4:7], 0 offen
    v_add_u32_e32 v42, 32, v40
    buffer_store_dwordx4 v[8:11], v42, s[4:7], 0 offen
    v_add_u32_e32 v42, 48, v40
    buffer_store_dwordx4 v[12:15], v42, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter15fwd_fp8_qk_tiledE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter15fwd_fp8_qk_tiledE
    .symbol: _ZN5aiter15fwd_fp8_qk_tiledE.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 80
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 24, .value_kind: by_value}
      - {.name: q_tile_idx, .size: 4, .offset: 28, .value_kind: by_value}
...
.end_amdgpu_metadata

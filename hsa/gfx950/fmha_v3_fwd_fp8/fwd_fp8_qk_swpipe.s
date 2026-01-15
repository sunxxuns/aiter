// FP8 QK with software-pipelined inner loop
// 256 threads (4 waves), overlaps ds_read with MFMA
// Key optimization: issue next iteration's reads while MFMA executes

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136
.set Q_LDS, 0                       // Q: 32*136 = 4352 bytes
.set K_LDS_A, 4352                  // K ping buffer
.set K_LDS_B, 8704                  // K pong buffer
.set LDS_SIZE, 16384

.text
.globl _ZN5aiter16fwd_fp8_qk_swpipeE
.p2align 8
.type _ZN5aiter16fwd_fp8_qk_swpipeE,@function

_ZN5aiter16fwd_fp8_qk_swpipeE:
    s_mov_b64 exec, -1
    
    // Args: O_ptr, Q_ptr, K_ptr, seq_len, q_tile_idx
    s_load_dwordx2 s[4:5], s[0:1], 0
    s_load_dwordx2 s[8:9], s[0:1], 8
    s_load_dwordx2 s[12:13], s[0:1], 16
    s_load_dword s20, s[0:1], 24
    s_load_dword s21, s[0:1], 28
    s_waitcnt lgkmcnt(0)
    
    // Buffer descriptors
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    v_and_b32_e32 v60, 63, v0
    v_lshrrev_b32_e32 v61, 6, v0
    v_and_b32_e32 v62, 255, v0
    
    // Load offsets for global memory
    v_lshrrev_b32_e32 v1, 3, v62
    v_and_b32_e32 v2, 7, v62
    v_lshlrev_b32_e32 v3, 7, v1
    v_lshlrev_b32_e32 v4, 4, v2
    v_add_u32_e32 v50, v3, v4
    
    // LDS write offset with pitch
    v_mov_b32_e32 v5, PITCH
    v_mul_lo_u32 v51, v1, v5
    v_add_u32_e32 v51, v51, v4
    
    // Load Q
    s_lshl_b32 s22, s21, 12
    v_add_u32_e32 v52, s22, v50
    buffer_load_dwordx4 v[20:23], v52, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v53, Q_LDS, v51
    ds_write_b128 v53, v[20:23]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // MFMA read addresses
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v63, v3, v4
    
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc
    
    v_mov_b32_e32 v1, PITCH
    v_mul_lo_u32 v70, v63, v1
    v_add_u32_e32 v70, v70, v64
    v_add_u32_e32 v70, Q_LDS, v70
    
    // num_k_tiles
    s_add_u32 s23, s20, 31
    s_lshr_b32 s23, s23, 5
    
    // Clear accumulators
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // PIPELINED K-LOOP with software-pipelined inner MFMA
    // ========================================================================
    
    // Load K tile 0 to buffer A
    s_mov_b32 s24, 0
    v_add_u32_e32 v54, 0, v50
    buffer_load_dwordx4 v[24:27], v54, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v55, K_LDS_A, v51
    ds_write_b128 v55, v[24:27]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_cmp_eq_u32 s23, 1
    s_cbranch_scc1 LAST_TILE
    
    s_mov_b32 s24, 1

PIPELINE_LOOP:
    // Load next K tile to buffer B (async)
    s_lshl_b32 s25, s24, 12
    v_add_u32_e32 v54, s25, v50
    buffer_load_dwordx4 v[28:31], v54, s[12:15], 0 offen
    
    // ====== SOFTWARE PIPELINED MFMA (buffer A) ======
    // Compute K_base address
    v_add_u32_e32 v71, K_LDS_A - Q_LDS, v70
    
    // Prolog: issue first Q/K reads
    ds_read_b64 v[32:33], v70 offset:0      // Q[0]
    v_add_u32_e32 v73, 0, v71
    ds_read_b64 v[34:35], v73               // K[0]
    
    // Iteration 1: issue reads + compute iter 0
    ds_read_b64 v[36:37], v70 offset:16     // Q[1]
    v_add_u32_e32 v73, 16, v71
    ds_read_b64 v[38:39], v73               // K[1]
    s_waitcnt lgkmcnt(2)                    // wait for Q[0],K[0]
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Iteration 2: issue reads + compute iter 1
    ds_read_b64 v[32:33], v70 offset:32     // Q[2]
    v_add_u32_e32 v73, 32, v71
    ds_read_b64 v[34:35], v73               // K[2]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Iteration 3: issue reads + compute iter 2
    ds_read_b64 v[36:37], v70 offset:48     // Q[3]
    v_add_u32_e32 v73, 48, v71
    ds_read_b64 v[38:39], v73               // K[3]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Iteration 4: issue reads + compute iter 3
    ds_read_b64 v[32:33], v70 offset:64     // Q[4]
    v_add_u32_e32 v73, 64, v71
    ds_read_b64 v[34:35], v73               // K[4]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Iteration 5: issue reads + compute iter 4
    ds_read_b64 v[36:37], v70 offset:80     // Q[5]
    v_add_u32_e32 v73, 80, v71
    ds_read_b64 v[38:39], v73               // K[5]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Iteration 6: issue reads + compute iter 5
    ds_read_b64 v[32:33], v70 offset:96     // Q[6]
    v_add_u32_e32 v73, 96, v71
    ds_read_b64 v[34:35], v73               // K[6]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Iteration 7: issue reads + compute iter 6
    ds_read_b64 v[36:37], v70 offset:112    // Q[7]
    v_add_u32_e32 v73, 112, v71
    ds_read_b64 v[38:39], v73               // K[7]
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Epilog: compute iter 7
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Wait for K load to buffer B
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v55, K_LDS_B, v51
    ds_write_b128 v55, v[28:31]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s24, s24, 1
    s_cmp_ge_u32 s24, s23
    s_cbranch_scc1 LAST_TILE_B
    
    // Load next K tile to buffer A (async)
    s_lshl_b32 s25, s24, 12
    v_add_u32_e32 v54, s25, v50
    buffer_load_dwordx4 v[24:27], v54, s[12:15], 0 offen
    
    // ====== SOFTWARE PIPELINED MFMA (buffer B) ======
    v_add_u32_e32 v71, K_LDS_B - Q_LDS, v70
    
    // Prolog
    ds_read_b64 v[32:33], v70 offset:0
    v_add_u32_e32 v73, 0, v71
    ds_read_b64 v[34:35], v73
    
    // Iter 1
    ds_read_b64 v[36:37], v70 offset:16
    v_add_u32_e32 v73, 16, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Iter 2
    ds_read_b64 v[32:33], v70 offset:32
    v_add_u32_e32 v73, 32, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Iter 3
    ds_read_b64 v[36:37], v70 offset:48
    v_add_u32_e32 v73, 48, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Iter 4
    ds_read_b64 v[32:33], v70 offset:64
    v_add_u32_e32 v73, 64, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Iter 5
    ds_read_b64 v[36:37], v70 offset:80
    v_add_u32_e32 v73, 80, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Iter 6
    ds_read_b64 v[32:33], v70 offset:96
    v_add_u32_e32 v73, 96, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Iter 7
    ds_read_b64 v[36:37], v70 offset:112
    v_add_u32_e32 v73, 112, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    // Epilog
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    // Wait for K load to buffer A
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v55, K_LDS_A, v51
    ds_write_b128 v55, v[24:27]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s24, s24, 1
    s_cmp_lt_u32 s24, s23
    s_cbranch_scc1 PIPELINE_LOOP
    
LAST_TILE:
    // Process last tile in buffer A (software pipelined)
    v_add_u32_e32 v71, K_LDS_A - Q_LDS, v70
    
    ds_read_b64 v[32:33], v70 offset:0
    v_add_u32_e32 v73, 0, v71
    ds_read_b64 v[34:35], v73
    
    ds_read_b64 v[36:37], v70 offset:16
    v_add_u32_e32 v73, 16, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    ds_read_b64 v[32:33], v70 offset:32
    v_add_u32_e32 v73, 32, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    ds_read_b64 v[36:37], v70 offset:48
    v_add_u32_e32 v73, 48, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    ds_read_b64 v[32:33], v70 offset:64
    v_add_u32_e32 v73, 64, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    ds_read_b64 v[36:37], v70 offset:80
    v_add_u32_e32 v73, 80, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    ds_read_b64 v[32:33], v70 offset:96
    v_add_u32_e32 v73, 96, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    ds_read_b64 v[36:37], v70 offset:112
    v_add_u32_e32 v73, 112, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    s_branch STORE_OUTPUT

LAST_TILE_B:
    // Process last tile in buffer B (software pipelined)
    v_add_u32_e32 v71, K_LDS_B - Q_LDS, v70
    
    ds_read_b64 v[32:33], v70 offset:0
    v_add_u32_e32 v73, 0, v71
    ds_read_b64 v[34:35], v73
    
    ds_read_b64 v[36:37], v70 offset:16
    v_add_u32_e32 v73, 16, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    ds_read_b64 v[32:33], v70 offset:32
    v_add_u32_e32 v73, 32, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    ds_read_b64 v[36:37], v70 offset:48
    v_add_u32_e32 v73, 48, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    ds_read_b64 v[32:33], v70 offset:64
    v_add_u32_e32 v73, 64, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    ds_read_b64 v[36:37], v70 offset:80
    v_add_u32_e32 v73, 80, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    ds_read_b64 v[32:33], v70 offset:96
    v_add_u32_e32 v73, 96, v71
    ds_read_b64 v[34:35], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
    ds_read_b64 v[36:37], v70 offset:112
    v_add_u32_e32 v73, 112, v71
    ds_read_b64 v[38:39], v73
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[32:33], v[34:35], v[0:15]
    
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[36:37], v[38:39], v[0:15]
    
STORE_OUTPUT:
    v_lshlrev_b32_e32 v40, 6, v60
    
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
.amdhsa_kernel _ZN5aiter16fwd_fp8_qk_swpipeE
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
  - .name: _ZN5aiter16fwd_fp8_qk_swpipeE
    .symbol: _ZN5aiter16fwd_fp8_qk_swpipeE.kd
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

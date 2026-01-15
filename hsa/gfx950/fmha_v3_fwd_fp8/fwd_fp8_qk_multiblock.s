// FP8 QK kernel with multi-block support
// Uses workgroup_id_x (s2) for Q tile indexing
// Uses workgroup_id_y (s3) for head indexing
// 256 threads (4 waves), pitch-136 LDS layout

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136
.set Q_LDS, 0
.set K_LDS_A, 4352
.set K_LDS_B, 8704
.set LDS_SIZE, 16384

// Kernel arguments layout:
// 0:  O_ptr (8 bytes)
// 8:  Q_ptr (8 bytes)
// 16: K_ptr (8 bytes)
// 24: seq_len (4 bytes)
// 28: head_dim (4 bytes) - should be 128
// 32: Q_stride_seq (4 bytes) - bytes between Q tiles (32 * 128 = 4096 for fp8)
// 36: Q_stride_head (4 bytes) - bytes between heads
// 40: K_stride_head (4 bytes) - bytes between heads for K
// 44: O_stride_seq (4 bytes) - bytes between output tiles (32 * seq_len * 4 for fp32)
// 48: O_stride_head (4 bytes) - bytes between heads for output

.text
.globl _ZN5aiter19fwd_fp8_qk_multiblockE
.p2align 8
.type _ZN5aiter19fwd_fp8_qk_multiblockE,@function

_ZN5aiter19fwd_fp8_qk_multiblockE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // SGPR LAYOUT:
    // s[0:1] = kernarg_segment_ptr (user SGPR)
    // s2 = workgroup_id_x = Q tile index
    // s3 = workgroup_id_y = head index  
    // s4 = workgroup_id_z = batch index (unused for now)
    // 
    // After loading:
    // s[16:19] = O buffer descriptor
    // s[20:23] = Q buffer descriptor
    // s[24:27] = K buffer descriptor
    // s[28:31] = strides and temps
    // ========================================================================
    
    // Save workgroup IDs before they get overwritten
    s_mov_b32 s5, s2                      // s5 = block_x (Q tile index)
    s_mov_b32 s6, s3                      // s6 = block_y (head index)
    
    // Load kernel arguments
    s_load_dwordx2 s[16:17], s[0:1], 0    // O_ptr
    s_load_dwordx2 s[20:21], s[0:1], 8    // Q_ptr
    s_load_dwordx2 s[24:25], s[0:1], 16   // K_ptr
    s_load_dword s28, s[0:1], 24          // seq_len
    s_load_dword s29, s[0:1], 28          // head_dim (128)
    s_load_dword s30, s[0:1], 32          // Q_stride_seq
    s_load_dword s31, s[0:1], 36          // Q_stride_head
    s_load_dword s32, s[0:1], 40          // K_stride_head
    s_load_dword s33, s[0:1], 44          // O_stride_seq
    s_load_dword s34, s[0:1], 48          // O_stride_head
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // MULTI-BLOCK: Apply workgroup offsets to base pointers
    // s5 = block_x = Q tile index
    // s6 = block_y = head index
    // ========================================================================
    
    // Q offset = block_x * Q_stride_seq + head * Q_stride_head
    s_mul_i32 s7, s5, s30           // s7 = block_x * Q_stride_seq
    s_mul_i32 s8, s6, s31           // s8 = head * Q_stride_head
    s_add_u32 s7, s7, s8            // s7 = total Q offset
    s_add_u32 s20, s20, s7          // Q_ptr += offset
    s_addc_u32 s21, s21, 0          // handle carry
    
    // K offset = head * K_stride_head (K is shared across Q tiles)
    s_mul_i32 s7, s6, s32           // s7 = head * K_stride_head
    s_add_u32 s24, s24, s7          // K_ptr += offset
    s_addc_u32 s25, s25, 0          // handle carry
    
    // O offset = block_x * O_stride_seq + head * O_stride_head
    s_mul_i32 s7, s5, s33           // s7 = block_x * O_stride_seq
    s_mul_i32 s8, s6, s34           // s8 = head * O_stride_head
    s_add_u32 s7, s7, s8            // s7 = total O offset
    s_add_u32 s16, s16, s7          // O_ptr += offset
    s_addc_u32 s17, s17, 0          // handle carry
    
    // ========================================================================
    // Set up buffer descriptors
    // s[16:19] = O, s[20:23] = Q, s[24:27] = K
    // ========================================================================
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000
    s_mov_b32 s22, -1
    s_mov_b32 s23, 0x20000
    s_mov_b32 s26, -1
    s_mov_b32 s27, 0x20000
    
    // ========================================================================
    // Thread indexing
    // ========================================================================
    v_and_b32_e32 v60, 63, v0             // lane_id (0-63)
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id (0-3)
    v_and_b32_e32 v62, 255, v0            // thread_id (0-255)
    
    // Global load offsets: each thread loads 16 bytes
    // Thread pattern: row = tid / 8, col_group = tid % 8
    v_lshrrev_b32_e32 v1, 3, v62          // row = tid / 8 (0-31)
    v_and_b32_e32 v2, 7, v62              // col_group = tid % 8 (0-7)
    v_lshlrev_b32_e32 v3, 7, v1           // row * 128
    v_lshlrev_b32_e32 v4, 4, v2           // col_group * 16
    v_add_u32_e32 v50, v3, v4             // global offset
    
    // LDS offsets with pitch-136
    v_mov_b32_e32 v5, PITCH
    v_mul_lo_u32 v51, v1, v5              // row * PITCH
    v_add_u32_e32 v51, v51, v4            // + col offset
    
    // ========================================================================
    // Load Q tile to LDS (using s[20:23] Q buffer descriptor)
    // ========================================================================
    buffer_load_dwordx4 v[80:83], v50, s[20:23], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v53, Q_LDS, v51
    ds_write_b128 v53, v[80:83]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // MFMA read address calculation
    // ========================================================================
    v_and_b32_e32 v1, 3, v60              // lane % 4
    v_lshrrev_b32_e32 v2, 3, v60          // lane / 8
    v_and_b32_e32 v2, 3, v2               // (lane / 8) % 4
    v_lshlrev_b32_e32 v2, 2, v2           // * 4
    v_add_u32_e32 v3, v1, v2              // base row
    v_lshrrev_b32_e32 v4, 2, v60          // lane / 4
    v_and_b32_e32 v4, 1, v4               // (lane / 4) % 2
    v_lshlrev_b32_e32 v4, 4, v4           // * 16
    v_add_u32_e32 v63, v3, v4             // row index
    
    // Column offset based on lane position
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc      // col offset: 0 or 8
    
    // Q read base address
    v_mov_b32_e32 v1, PITCH
    v_mul_lo_u32 v70, v63, v1             // row * PITCH
    v_add_u32_e32 v70, v70, v64           // + col offset
    v_add_u32_e32 v70, Q_LDS, v70         // + Q_LDS base
    
    // K read base (relative to Q)
    v_add_u32_e32 v71, K_LDS_A - Q_LDS, v70
    
    // ========================================================================
    // Calculate num_k_tiles (seq_len is in s28)
    // ========================================================================
    s_add_u32 s35, s28, 31
    s_lshr_b32 s35, s35, 5                // s35 = num_k_tiles = (seq_len + 31) / 32
    
    // ========================================================================
    // Clear accumulators
    // ========================================================================
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // PRE-LOAD Q chunks (reused for all K tiles)
    // ========================================================================
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
    // Load K tile 0 to buffer A (using s[24:27] K buffer descriptor)
    // ========================================================================
    s_mov_b32 s36, 0                      // k_tile_idx
    buffer_load_dwordx4 v[80:83], v50, s[24:27], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v55, K_LDS_A, v51
    ds_write_b128 v55, v[80:83]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Check if only 1 tile
    s_cmp_eq_u32 s35, 1
    s_cbranch_scc1 LAST_TILE
    
    s_mov_b32 s36, 1

PIPELINE_LOOP:
    // Start async load of next K tile to buffer B
    s_lshl_b32 s37, s36, 12               // k_tile_idx * 4096
    v_add_u32_e32 v54, s37, v50
    buffer_load_dwordx4 v[84:87], v54, s[24:27], 0 offen
    
    // PRE-LOAD K chunks from buffer A
    ds_read_b64 v[32:33], v71 offset:0
    ds_read_b64 v[34:35], v71 offset:16
    ds_read_b64 v[36:37], v71 offset:32
    ds_read_b64 v[38:39], v71 offset:48
    ds_read_b64 v[40:41], v71 offset:64
    ds_read_b64 v[42:43], v71 offset:80
    ds_read_b64 v[44:45], v71 offset:96
    ds_read_b64 v[46:47], v71 offset:112
    s_waitcnt lgkmcnt(0)
    
    // CHAIN ALL 8 MFMAs
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
    v_add_u32_e32 v55, K_LDS_B, v51
    ds_write_b128 v55, v[84:87]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s36, s36, 1
    s_cmp_ge_u32 s36, s35
    s_cbranch_scc1 LAST_TILE_B
    
    // Start async load of next K tile to buffer A
    s_lshl_b32 s37, s36, 12
    v_add_u32_e32 v54, s37, v50
    buffer_load_dwordx4 v[80:83], v54, s[24:27], 0 offen
    
    // K base for buffer B
    v_add_u32_e32 v72, K_LDS_B - Q_LDS, v70
    
    // PRE-LOAD K chunks from buffer B
    ds_read_b64 v[32:33], v72 offset:0
    ds_read_b64 v[34:35], v72 offset:16
    ds_read_b64 v[36:37], v72 offset:32
    ds_read_b64 v[38:39], v72 offset:48
    ds_read_b64 v[40:41], v72 offset:64
    ds_read_b64 v[42:43], v72 offset:80
    ds_read_b64 v[44:45], v72 offset:96
    ds_read_b64 v[46:47], v72 offset:112
    s_waitcnt lgkmcnt(0)
    
    // CHAIN ALL 8 MFMAs
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
    v_add_u32_e32 v55, K_LDS_A, v51
    ds_write_b128 v55, v[80:83]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_add_u32 s36, s36, 1
    s_cmp_lt_u32 s36, s35
    s_cbranch_scc1 PIPELINE_LOOP

LAST_TILE:
    // Process last tile in buffer A
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
    // Process last tile in buffer B
    v_add_u32_e32 v72, K_LDS_B - Q_LDS, v70
    
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
    // Store 32x32 fp32 output (using s[16:19] O buffer descriptor)
    // Each thread stores 16 floats (64 bytes) at its unique location
    // Output layout: contiguous 32x32 matrix per Q tile
    
    // Calculate output offset based on thread position
    // MFMA 32x32x16 output: 16 fp32 values per thread arranged as 4x4 blocks
    // Thread mapping follows MFMA accumulator layout
    
    // For now: simple contiguous store (each wave writes to different region)
    // Wave 0: rows 0-7, Wave 1: rows 8-15, Wave 2: rows 16-23, Wave 3: rows 24-31
    
    v_lshlrev_b32_e32 v48, 6, v62         // thread_id * 64 bytes
    
    buffer_store_dwordx4 v[0:3], v48, s[16:19], 0 offen
    v_add_u32_e32 v49, 16, v48
    buffer_store_dwordx4 v[4:7], v49, s[16:19], 0 offen
    v_add_u32_e32 v49, 32, v48
    buffer_store_dwordx4 v[8:11], v49, s[16:19], 0 offen
    v_add_u32_e32 v49, 48, v48
    buffer_store_dwordx4 v[12:15], v49, s[16:19], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter19fwd_fp8_qk_multiblockE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 64
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 96
    .amdhsa_next_free_sgpr 40
    .amdhsa_accum_offset 96
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter19fwd_fp8_qk_multiblockE
    .symbol: _ZN5aiter19fwd_fp8_qk_multiblockE.kd
    .kernarg_segment_size: 64
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 40
    .vgpr_count: 96
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 24, .value_kind: by_value}
      - {.name: head_dim, .size: 4, .offset: 28, .value_kind: by_value}
      - {.name: Q_stride_seq, .size: 4, .offset: 32, .value_kind: by_value}
      - {.name: Q_stride_head, .size: 4, .offset: 36, .value_kind: by_value}
      - {.name: K_stride_head, .size: 4, .offset: 40, .value_kind: by_value}
      - {.name: O_stride_seq, .size: 4, .offset: 44, .value_kind: by_value}
      - {.name: O_stride_head, .size: 4, .offset: 48, .value_kind: by_value}
...
.end_amdgpu_metadata

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set Q_LDS, 0                 // 128×128 (TR8 interleaved, 4×4096)
.set K_LDS, 16384             // 32×128 (TR8 interleaved)
.set LDS_SIZE, 20480          // Q + K

.text
.globl _fwd_fp8_qk_debug
.p2align 8
.type _fwd_fp8_qk_debug,@function

_fwd_fp8_qk_debug:
    s_mov_b64 exec, -1

    // ------------------------------------------------------------------------
    // Load kernel arguments
    // ------------------------------------------------------------------------
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr (float32)
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr (fp8)
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr (fp8)
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Buffer descriptors (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000

    // ------------------------------------------------------------------------
    // Thread indexing (256 threads)
    // ------------------------------------------------------------------------
    v_mov_b32_e32 v60, v0                // tid (0-255)
    v_lshrrev_b32_e32 v9, 6, v60         // wave_id = tid / 64 (0-3)
    v_and_b32_e32 v10, 63, v60           // lane_id = tid % 64

    // ------------------------------------------------------------------------
    // Load Q tile to LDS (TR8 interleaved layout)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v1, 6, v60         // tid * 64 bytes
    v_mov_b32_e32 v2, v1                 // Q global offset

    buffer_load_dwordx4 v[40:43], v2, s[8:11], 0 offen
    v_add_u32_e32 v3, 16, v2
    buffer_load_dwordx4 v[44:47], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 32, v2
    buffer_load_dwordx4 v[48:51], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 48, v2
    buffer_load_dwordx4 v[52:55], v3, s[8:11], 0 offen
    s_waitcnt vmcnt(0)

    // row_global = tid >> 1, row_in_wave = row_global & 31
    v_lshrrev_b32_e32 v30, 1, v60        // row_global
    v_and_b32_e32 v31, 31, v30           // row_in_wave
    v_and_b32_e32 v32, 7, v31            // row_in_block
    v_lshrrev_b32_e32 v33, 3, v31        // block
    v_lshlrev_b32_e32 v33, 10, v33       // block_base = block * 1024
    v_and_b32_e32 v34, 1, v60
    v_lshlrev_b32_e32 v34, 9, v34        // col_base * 8 (0 or 512)
    v_lshlrev_b32_e32 v35, 12, v9        // wave_base = wave_id * 4096
    v_add_u32_e32 v36, v33, v32
    v_add_u32_e32 v36, v36, v35
    v_add_u32_e32 v36, v36, v34
    v_add_u32_e32 v36, Q_LDS, v36        // base_q

    // Scatter 64 bytes with stride 8 (4 chunks)
    v_mov_b32_e32 v37, v36
    // chunk 0 (v40-v43)
    v_bfe_u32 v24, v40, 0, 8
    ds_write_b8 v37, v24
    v_add_u32_e32 v25, 8, v37
    v_bfe_u32 v24, v40, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 16, v37
    v_bfe_u32 v24, v40, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 24, v37
    v_bfe_u32 v24, v40, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 32, v37
    v_bfe_u32 v24, v41, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 40, v37
    v_bfe_u32 v24, v41, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 48, v37
    v_bfe_u32 v24, v41, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 56, v37
    v_bfe_u32 v24, v41, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 64, v37
    v_bfe_u32 v24, v42, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 72, v37
    v_bfe_u32 v24, v42, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 80, v37
    v_bfe_u32 v24, v42, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 88, v37
    v_bfe_u32 v24, v42, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 96, v37
    v_bfe_u32 v24, v43, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 104, v37
    v_bfe_u32 v24, v43, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 112, v37
    v_bfe_u32 v24, v43, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 120, v37
    v_bfe_u32 v24, v43, 24, 8
    ds_write_b8 v25, v24

    // chunk 1 (v44-v47)
    v_add_u32_e32 v37, 128, v36
    v_bfe_u32 v24, v44, 0, 8
    ds_write_b8 v37, v24
    v_add_u32_e32 v25, 8, v37
    v_bfe_u32 v24, v44, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 16, v37
    v_bfe_u32 v24, v44, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 24, v37
    v_bfe_u32 v24, v44, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 32, v37
    v_bfe_u32 v24, v45, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 40, v37
    v_bfe_u32 v24, v45, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 48, v37
    v_bfe_u32 v24, v45, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 56, v37
    v_bfe_u32 v24, v45, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 64, v37
    v_bfe_u32 v24, v46, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 72, v37
    v_bfe_u32 v24, v46, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 80, v37
    v_bfe_u32 v24, v46, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 88, v37
    v_bfe_u32 v24, v46, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 96, v37
    v_bfe_u32 v24, v47, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 104, v37
    v_bfe_u32 v24, v47, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 112, v37
    v_bfe_u32 v24, v47, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 120, v37
    v_bfe_u32 v24, v47, 24, 8
    ds_write_b8 v25, v24

    // chunk 2 (v48-v51)
    v_add_u32_e32 v37, 256, v36
    v_bfe_u32 v24, v48, 0, 8
    ds_write_b8 v37, v24
    v_add_u32_e32 v25, 8, v37
    v_bfe_u32 v24, v48, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 16, v37
    v_bfe_u32 v24, v48, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 24, v37
    v_bfe_u32 v24, v48, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 32, v37
    v_bfe_u32 v24, v49, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 40, v37
    v_bfe_u32 v24, v49, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 48, v37
    v_bfe_u32 v24, v49, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 56, v37
    v_bfe_u32 v24, v49, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 64, v37
    v_bfe_u32 v24, v50, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 72, v37
    v_bfe_u32 v24, v50, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 80, v37
    v_bfe_u32 v24, v50, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 88, v37
    v_bfe_u32 v24, v50, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 96, v37
    v_bfe_u32 v24, v51, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 104, v37
    v_bfe_u32 v24, v51, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 112, v37
    v_bfe_u32 v24, v51, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 120, v37
    v_bfe_u32 v24, v51, 24, 8
    ds_write_b8 v25, v24

    // chunk 3 (v52-v55)
    v_add_u32_e32 v37, 384, v36
    v_bfe_u32 v24, v52, 0, 8
    ds_write_b8 v37, v24
    v_add_u32_e32 v25, 8, v37
    v_bfe_u32 v24, v52, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 16, v37
    v_bfe_u32 v24, v52, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 24, v37
    v_bfe_u32 v24, v52, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 32, v37
    v_bfe_u32 v24, v53, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 40, v37
    v_bfe_u32 v24, v53, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 48, v37
    v_bfe_u32 v24, v53, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 56, v37
    v_bfe_u32 v24, v53, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 64, v37
    v_bfe_u32 v24, v54, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 72, v37
    v_bfe_u32 v24, v54, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 80, v37
    v_bfe_u32 v24, v54, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 88, v37
    v_bfe_u32 v24, v54, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 96, v37
    v_bfe_u32 v24, v55, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 104, v37
    v_bfe_u32 v24, v55, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 112, v37
    v_bfe_u32 v24, v55, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 120, v37
    v_bfe_u32 v24, v55, 24, 8
    ds_write_b8 v25, v24

    // ------------------------------------------------------------------------
    // Load K tile to LDS (TR8 interleaved layout)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v13, 4, v60        // tid * 16 bytes
    v_mov_b32_e32 v14, v13               // K global offset
    buffer_load_dwordx4 v[20:23], v14, s[12:15], 0 offen
    s_waitcnt vmcnt(0)

    // row = tid / 8, k_chunk = tid % 8
    v_lshrrev_b32_e32 v15, 3, v60        // row
    v_and_b32_e32 v16, 7, v60            // k_chunk
    v_and_b32_e32 v17, 7, v15            // row_in_block = row % 8
    v_lshrrev_b32_e32 v18, 3, v15        // block = row / 8
    v_lshlrev_b32_e32 v18, 10, v18       // block_base = block * 1024
    v_lshlrev_b32_e32 v19, 7, v16        // k_base = k_chunk * 128
    v_add_u32_e32 v19, v19, v17
    v_add_u32_e32 v19, v19, v18          // base = row_in_block + block_base + k_base
    v_add_u32_e32 v19, K_LDS, v19

    // Scatter 16 bytes with stride 8
    v_bfe_u32 v24, v20, 0, 8
    ds_write_b8 v19, v24
    v_add_u32_e32 v25, 8, v19
    v_bfe_u32 v24, v20, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 16, v19
    v_bfe_u32 v24, v20, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 24, v19
    v_bfe_u32 v24, v20, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 32, v19
    v_bfe_u32 v24, v21, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 40, v19
    v_bfe_u32 v24, v21, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 48, v19
    v_bfe_u32 v24, v21, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 56, v19
    v_bfe_u32 v24, v21, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 64, v19
    v_bfe_u32 v24, v22, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 72, v19
    v_bfe_u32 v24, v22, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 80, v19
    v_bfe_u32 v24, v22, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 88, v19
    v_bfe_u32 v24, v22, 24, 8
    ds_write_b8 v25, v24

    v_add_u32_e32 v25, 96, v19
    v_bfe_u32 v24, v23, 0, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 104, v19
    v_bfe_u32 v24, v23, 8, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 112, v19
    v_bfe_u32 v24, v23, 16, 8
    ds_write_b8 v25, v24
    v_add_u32_e32 v25, 120, v19
    v_bfe_u32 v24, v23, 24, 8
    ds_write_b8 v25, v24

    s_waitcnt lgkmcnt(0)
    s_barrier

    // ------------------------------------------------------------------------
    // Compute MFMA LDS read addresses (TR8 layout)
    // ------------------------------------------------------------------------
    v_and_b32_e32 v11, 15, v10           // lane & 15
    v_lshrrev_b32_e32 v12, 4, v10
    v_and_b32_e32 v12, 1, v12
    v_lshlrev_b32_e32 v12, 4, v12
    v_add_u32_e32 v13, v11, v12          // mfma_row

    v_and_b32_e32 v14, 7, v13            // row_in_block
    v_lshrrev_b32_e32 v15, 3, v13        // block = row / 8
    v_lshlrev_b32_e32 v15, 10, v15       // block_base = block * 1024
    v_lshlrev_b32_e32 v17, 12, v9        // wave_base = wave_id * 4096
    v_add_u32_e32 v18, v15, v14
    v_add_u32_e32 v24, v18, v17
    v_add_u32_e32 v24, Q_LDS, v24        // base for Q

    v_add_u32_e32 v27, K_LDS, v18        // base for K

    // ------------------------------------------------------------------------
    // QK MFMA (K=64 × 2)
    // ------------------------------------------------------------------------
    .irp i, 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
        v_mov_b32_e32 v\i, 0
    .endr

    ds_read_b64_tr_b8 v[0:1], v24 offset:0
    ds_read_b64_tr_b8 v[2:3], v24 offset:64
    ds_read_b64_tr_b8 v[4:5], v24 offset:128
    ds_read_b64_tr_b8 v[6:7], v24 offset:192
    ds_read_b64_tr_b8 v[16:17], v27 offset:0
    ds_read_b64_tr_b8 v[18:19], v27 offset:64
    ds_read_b64_tr_b8 v[20:21], v27 offset:128
    ds_read_b64_tr_b8 v[22:23], v27 offset:192
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    v_add_u32_e32 v28, 512, v24
    v_add_u32_e32 v29, 512, v27
    ds_read_b64_tr_b8 v[0:1], v28 offset:0
    ds_read_b64_tr_b8 v[2:3], v28 offset:64
    ds_read_b64_tr_b8 v[4:5], v28 offset:128
    ds_read_b64_tr_b8 v[6:7], v28 offset:192
    ds_read_b64_tr_b8 v[16:17], v29 offset:0
    ds_read_b64_tr_b8 v[18:19], v29 offset:64
    ds_read_b64_tr_b8 v[20:21], v29 offset:128
    ds_read_b64_tr_b8 v[22:23], v29 offset:192
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    // ------------------------------------------------------------------------
    // Store QK output (16 floats per thread)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v50, 6, v60        // tid * 64 bytes
    v_mov_b32_e32 v51, v50

    buffer_store_dwordx4 v[32:35], v51, s[4:7], 0 offen
    v_add_u32_e32 v52, 16, v51
    buffer_store_dwordx4 v[36:39], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 32, v51
    buffer_store_dwordx4 v[40:43], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 48, v51
    buffer_store_dwordx4 v[44:47], v52, s[4:7], 0 offen

    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_qk_debug
    .amdhsa_group_segment_fixed_size 20480
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 220
    .amdhsa_next_free_sgpr 30
    .amdhsa_accum_offset 220
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_qk_debug
    .symbol: _fwd_fp8_qk_debug.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 20480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 30
    .vgpr_count: 220
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

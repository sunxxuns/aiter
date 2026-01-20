.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set Q_LDS, 0                 // 128×128 (pitch-132)
.set K_LDS, 16896             // 32×128 (row-major)
.set LDS_SIZE, 24576          // Q + K (>= 20992)

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
    // Load Q tile to LDS (Triton swizzle)
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

    // Q LDS address (pitch-132): row = tid>>1, col = (tid&1)*64
    v_and_b32_e32 v11, 0xFF, v60         // tid_in_tile (0-255)
    v_lshrrev_b32_e32 v11, 1, v11        // row
    v_lshlrev_b32_e32 v12, 7, v11        // row * 128
    v_lshlrev_b32_e32 v13, 2, v11        // row * 4
    v_add_u32_e32 v12, v12, v13          // row * 132
    v_and_b32_e32 v11, 1, v60
    v_lshlrev_b32_e32 v11, 6, v11        // col offset
    v_add_u32_e32 v20, v12, v11
    v_add_u32_e32 v20, Q_LDS, v20

    ds_write_b128 v20, v[40:43]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[44:47]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[48:51]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[52:55]

    // ------------------------------------------------------------------------
    // Load K tile to LDS (row-major, 256 threads)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v13, 4, v60        // tid * 16 bytes
    buffer_load_dwordx4 v[20:23], v13, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v38, K_LDS, v13
    ds_write_b128 v38, v[20:23]

    s_waitcnt lgkmcnt(0)
    s_barrier

    // ------------------------------------------------------------------------
    // Compute MFMA LDS read addresses (pitch-132 Q, row-major K)
    // ------------------------------------------------------------------------
    v_and_b32_e32 v11, 15, v10           // lane & 15
    v_lshrrev_b32_e32 v12, 4, v10
    v_and_b32_e32 v12, 1, v12
    v_lshlrev_b32_e32 v12, 4, v12
    v_add_u32_e32 v13, v11, v12          // mfma_row (0..31)

    v_lshlrev_b32_e32 v14, 7, v13        // row * 128
    v_lshlrev_b32_e32 v15, 2, v13        // row * 4
    v_add_u32_e32 v14, v14, v15          // Q row offset (row * 132)
    v_lshlrev_b32_e32 v18, 7, v13        // K row offset (row * 128)

    v_cmp_ge_u32_e64 vcc, v10, 32
    v_cndmask_b32_e64 v11, 0, 16, vcc    // k_off1
    v_cndmask_b32_e64 v12, 32, 48, vcc   // k_off2

    v_lshrrev_b32_e32 v9, 6, v60         // wave_id = tid / 64
    v_and_b32_e32 v15, 3, v9             // wave_in_tile
    v_lshlrev_b32_e32 v17, 12, v15
    v_lshlrev_b32_e32 v19, 7, v15
    v_add_u32_e32 v17, v17, v19          // wave_in_tile * 4224

    v_add_u32_e32 v58, v17, v14          // Q base + row
    v_add_u32_e32 v24, v58, v11          // Q addr1
    v_add_u32_e32 v25, v58, v12          // Q addr2

    v_add_u32_e32 v26, K_LDS, v18        // K base + row
    v_add_u32_e32 v27, v26, v11          // K addr1
    v_add_u32_e32 v28, v26, v12          // K addr2

    // ------------------------------------------------------------------------
    // QK MFMA (K=64 × 2)
    // ------------------------------------------------------------------------
    .irp i, 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
        v_mov_b32_e32 v\i, 0
    .endr

    // First K=64 half
    ds_read_b128 v[0:3], v24
    ds_read_b128 v[4:7], v25
    ds_read_b128 v[16:19], v27
    ds_read_b128 v[20:23], v28
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    // Second K=64 half
    ds_read_b128 v[0:3], v24 offset:64
    ds_read_b128 v[4:7], v25 offset:64
    ds_read_b128 v[16:19], v27 offset:64
    ds_read_b128 v[20:23], v28 offset:64
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
    .amdhsa_group_segment_fixed_size 24576
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

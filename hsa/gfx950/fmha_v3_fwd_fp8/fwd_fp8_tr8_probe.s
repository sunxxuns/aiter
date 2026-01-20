.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set LDS_SIZE, 8192

.text
.globl _fwd_fp8_tr8_probe
.p2align 8
.type _fwd_fp8_tr8_probe,@function

_fwd_fp8_tr8_probe:
    s_mov_b64 exec, -1

    // ------------------------------------------------------------------------
    // Load kernel arguments
    // ------------------------------------------------------------------------
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr (uint8)
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr (uint8)
    s_load_dword s12, s[0:1], 16          // base_offset
    s_waitcnt lgkmcnt(0)

    // Buffer descriptors (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000

    // ------------------------------------------------------------------------
    // Thread indexing (256 threads)
    // ------------------------------------------------------------------------
    v_mov_b32_e32 v60, v0                // tid (0-255)

    // ------------------------------------------------------------------------
    // Load Q[8×128] to LDS in TR8 interleaved layout
    // ------------------------------------------------------------------------
    // row = tid / 8, k_chunk = tid % 8
    v_lshrrev_b32_e32 v10, 3, v60        // row
    v_and_b32_e32 v11, 7, v60            // k_chunk
    v_lshlrev_b32_e32 v12, 7, v10        // row * 128
    v_lshlrev_b32_e32 v13, 4, v11        // k_chunk * 16
    v_add_u32_e32 v14, v12, v13          // global offset

    buffer_load_dwordx4 v[0:3], v14, s[8:11], 0 offen
    s_waitcnt vmcnt(0)

    // LDS base = row + k_chunk*128 (stride 8 layout for 8 rows)
    v_lshlrev_b32_e32 v15, 7, v11        // k_chunk * 128
    v_add_u32_e32 v15, v10, v15          // base

    // Scatter 16 bytes with stride 8
    v_bfe_u32 v20, v0, 0, 8
    ds_write_b8 v15, v20
    v_add_u32_e32 v16, 8, v15
    v_bfe_u32 v20, v0, 8, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 16, v15
    v_bfe_u32 v20, v0, 16, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 24, v15
    v_bfe_u32 v20, v0, 24, 8
    ds_write_b8 v16, v20

    v_add_u32_e32 v16, 32, v15
    v_bfe_u32 v20, v1, 0, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 40, v15
    v_bfe_u32 v20, v1, 8, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 48, v15
    v_bfe_u32 v20, v1, 16, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 56, v15
    v_bfe_u32 v20, v1, 24, 8
    ds_write_b8 v16, v20

    v_add_u32_e32 v16, 64, v15
    v_bfe_u32 v20, v2, 0, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 72, v15
    v_bfe_u32 v20, v2, 8, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 80, v15
    v_bfe_u32 v20, v2, 16, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 88, v15
    v_bfe_u32 v20, v2, 24, 8
    ds_write_b8 v16, v20

    v_add_u32_e32 v16, 96, v15
    v_bfe_u32 v20, v3, 0, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 104, v15
    v_bfe_u32 v20, v3, 8, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 112, v15
    v_bfe_u32 v20, v3, 16, 8
    ds_write_b8 v16, v20
    v_add_u32_e32 v16, 120, v15
    v_bfe_u32 v20, v3, 24, 8
    ds_write_b8 v16, v20

    s_waitcnt lgkmcnt(0)
    s_barrier

    // ------------------------------------------------------------------------
    // Read back with TR8 (base = row)
    // ------------------------------------------------------------------------
    v_and_b32_e32 v30, 7, v60             // row = tid % 8
    v_mov_b32_e32 v31, s12
    v_add_u32_e32 v30, v30, v31
    ds_read_b64_tr_b8 v[40:41], v30
    s_waitcnt lgkmcnt(0)

    // ------------------------------------------------------------------------
    // Store 8 bytes per thread (first 64 threads)
    // ------------------------------------------------------------------------
    v_cmp_lt_u32_e64 vcc, v60, 64
    s_and_saveexec_b64 s[20:21], vcc
    v_lshlrev_b32_e32 v50, 3, v60         // tid * 8
    buffer_store_dwordx2 v[40:41], v50, s[4:7], 0 offen
    s_mov_b64 exec, s[20:21]

    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_tr8_probe
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_tr8_probe
    .symbol: _fwd_fp8_tr8_probe.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 64
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: base_offset, .size: 4, .offset: 16, .value_kind: by_value}
...
.end_amdgpu_metadata

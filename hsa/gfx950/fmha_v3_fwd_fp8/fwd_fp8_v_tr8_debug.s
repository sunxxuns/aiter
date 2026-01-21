// Debug kernel: load V into LDS (swizzled) and read via ds_read_b64_tr_b8
// Outputs 64 bytes per thread (8x ds_read_b64_tr_b8).
.text
.p2align 8
.globl _fwd_fp8_v_tr8_debug
.type _fwd_fp8_v_tr8_debug, @function

.set V_LDS0, 0
.set LDS_SIZE, 8192

_fwd_fp8_v_tr8_debug:
    s_mov_b64 exec, -1
    // Load kernel args (scaffold-compatible)
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr (unused)
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr (unused)
    s_load_dwordx2 s[16:17], s[0:1], 24   // V_ptr
    s_load_dword s20, s[0:1], 32          // num_k_tiles (unused)
    s_load_dword s21, s[0:1], 36          // stride_qh (unused)
    s_load_dword s22, s[0:1], 40          // stride_kh (unused)
    s_load_dword s23, s[0:1], 44          // stride_vh (unused)
    s_load_dword s24, s[0:1], 48          // stride_oh (unused)
    s_waitcnt lgkmcnt(0)

    // Buffer descriptors (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000

    v_mov_b32_e32 v60, v0                 // tid

    // Load V (tid < 256)
    v_mov_b32_e32 v2, 256
    v_cmp_lt_u32_e32 vcc, v60, v2
    s_and_saveexec_b64 s[8:9], vcc
    v_lshlrev_b32_e32 v2, 5, v60          // tid * 32 bytes
    buffer_load_dwordx4 v[40:43], v2, s[16:19], 0 offen
    v_add_u32_e32 v3, 16, v2
    buffer_load_dwordx4 v[44:47], v3, s[16:19], 0 offen
    s_waitcnt vmcnt(0)

    // Swizzled LDS write with byte toggle (debug probe)
    // Row-major LDS write
    v_lshrrev_b32_e32 v4, 2, v60         // row = tid >> 2
    v_and_b32_e32 v5, 3, v60             // col_block = tid & 3
    v_lshlrev_b32_e32 v4, 7, v4          // row * 128
    v_lshlrev_b32_e32 v5, 5, v5          // col_block * 32
    v_add_u32_e32 v4, v4, v5
    ds_write_b128 v4, v[40:43]
    v_mov_b32_e32 v7, 16
    v_add_u32_e32 v6, v4, v7
    ds_write_b128 v6, v[44:47]
    s_mov_b64 exec, s[8:9]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // TR8 base (same as scaffold)
    s_movk_i32 s25, 0xb80
    v_lshlrev_b32_e32 v2, 6, v60
    v_lshlrev_b32_e32 v3, 2, v60
    v_and_b32_e32 v4, 48, v3
    v_and_or_b32 v2, v2, s25, v4
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v2, v2, v5, v6 bitop3:0x36

    v_xor_b32_e32 v3, 0x20, v2
    v_xor_b32_e32 v4, 0x460, v2
    v_xor_b32_e32 v5, 0x1020, v2
    v_xor_b32_e32 v6, 0x1460, v2
    v_xor_b32_e32 v7, 0x60, v2
    v_xor_b32_e32 v8, 0x420, v2
    v_xor_b32_e32 v9, 0x1060, v2
    v_xor_b32_e32 v10, 0x1420, v2

    // Read 16x b64 (4 from base v2 with offsets, 4 from v3..v6,
    // then 4 from base v2 offset by 64, and 4 from v7..v10)
    ds_read_b64_tr_b8 v[0:1], v2 offset:0
    ds_read_b64_tr_b8 v[2:3], v2 offset:1088
    ds_read_b64_tr_b8 v[4:5], v2 offset:4096
    ds_read_b64_tr_b8 v[6:7], v2 offset:5184

    ds_read_b64_tr_b8 v[8:9], v3
    ds_read_b64_tr_b8 v[10:11], v4
    ds_read_b64_tr_b8 v[12:13], v5
    ds_read_b64_tr_b8 v[14:15], v6

    ds_read_b64_tr_b8 v[16:17], v2 offset:64
    ds_read_b64_tr_b8 v[18:19], v2 offset:1024
    ds_read_b64_tr_b8 v[20:21], v2 offset:4160
    ds_read_b64_tr_b8 v[22:23], v2 offset:5120

    ds_read_b64_tr_b8 v[24:25], v7
    ds_read_b64_tr_b8 v[26:27], v8
    ds_read_b64_tr_b8 v[28:29], v9
    ds_read_b64_tr_b8 v[30:31], v10
    s_waitcnt lgkmcnt(0)

    // Store 128 bytes per thread
    v_lshlrev_b32_e32 v32, 7, v60         // tid * 128 bytes
    buffer_store_dwordx4 v[0:3], v32, s[4:7], 0 offen
    v_add_u32_e32 v33, 16, v32
    buffer_store_dwordx4 v[4:7], v33, s[4:7], 0 offen
    v_add_u32_e32 v33, 32, v32
    buffer_store_dwordx4 v[8:11], v33, s[4:7], 0 offen
    v_add_u32_e32 v33, 48, v32
    buffer_store_dwordx4 v[12:15], v33, s[4:7], 0 offen
    v_add_u32_e32 v33, 64, v32
    buffer_store_dwordx4 v[16:19], v33, s[4:7], 0 offen
    v_add_u32_e32 v33, 80, v32
    buffer_store_dwordx4 v[20:23], v33, s[4:7], 0 offen
    v_add_u32_e32 v33, 96, v32
    buffer_store_dwordx4 v[24:27], v33, s[4:7], 0 offen
    v_add_u32_e32 v33, 112, v32
    buffer_store_dwordx4 v[28:31], v33, s[4:7], 0 offen

    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_v_tr8_debug
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 52
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_v_tr8_debug
    .symbol: _fwd_fp8_v_tr8_debug.kd
    .kernarg_segment_size: 52
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V_ptr, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: num_k_tiles, .size: 4, .offset: 32, .value_kind: by_value}
      - {.name: stride_qh, .size: 4, .offset: 36, .value_kind: by_value}
      - {.name: stride_kh, .size: 4, .offset: 40, .value_kind: by_value}
      - {.name: stride_vh, .size: 4, .offset: 44, .value_kind: by_value}
      - {.name: stride_oh, .size: 4, .offset: 48, .value_kind: by_value}
...
.end_amdgpu_metadata

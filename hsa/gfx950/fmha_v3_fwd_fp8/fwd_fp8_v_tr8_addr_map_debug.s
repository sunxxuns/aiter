// Debug kernel: fill LDS with known pattern, then TR8 read map.
.text
.p2align 8
.globl _fwd_fp8_v_tr8_addr_map_debug
.type _fwd_fp8_v_tr8_addr_map_debug, @function

.set LDS_SIZE, 8192

_fwd_fp8_v_tr8_addr_map_debug:
    s_mov_b64 exec, -1
    // Load kernel args (scaffold-compatible)
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr (unused)
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr (unused)
    s_load_dwordx2 s[16:17], s[0:1], 24   // V_ptr (pattern)
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

    // Load 16B pattern for [0..4095] and [4096..8191]
    v_lshlrev_b32_e32 v2, 4, v60          // tid * 16 bytes
    buffer_load_dwordx4 v[40:43], v2, s[16:19], 0 offen
    v_add_u32_e32 v3, 4096, v2
    buffer_load_dwordx4 v[44:47], v3, s[16:19], 0 offen
    s_waitcnt vmcnt(0)

    // Write pattern into LDS (no swizzle)
    ds_write_b128 v2, v[40:43]
    ds_write_b128 v2, v[44:47] offset:4096
    s_waitcnt lgkmcnt(0)
    s_barrier

    // TR8 base (same as scaffold)
    s_movk_i32 s25, 0xb80
    v_lshlrev_b32_e32 v20, 6, v60
    v_lshlrev_b32_e32 v21, 2, v60
    v_and_b32_e32 v22, 48, v21
    v_and_or_b32 v20, v20, s25, v22
    v_and_b32_e32 v23, 16, v60
    v_lshlrev_b32_e32 v24, 3, v60
    v_and_b32_e32 v24, 8, v24
    v_bitop3_b32 v20, v20, v23, v24 bitop3:0x36

    v_xor_b32_e32 v21, 0x20, v20
    v_xor_b32_e32 v22, 0x460, v20
    v_xor_b32_e32 v23, 0x1020, v20
    v_xor_b32_e32 v24, 0x1460, v20
    v_xor_b32_e32 v25, 0x60, v20
    v_xor_b32_e32 v26, 0x420, v20
    v_xor_b32_e32 v27, 0x1060, v20
    v_xor_b32_e32 v28, 0x1420, v20

    // Read 16x b64 (same order as scaffold)
    ds_read_b64_tr_b8 v[0:1], v20 offset:0
    ds_read_b64_tr_b8 v[2:3], v20 offset:1088
    ds_read_b64_tr_b8 v[4:5], v20 offset:4096
    ds_read_b64_tr_b8 v[6:7], v20 offset:5184

    ds_read_b64_tr_b8 v[8:9], v21
    ds_read_b64_tr_b8 v[10:11], v22
    ds_read_b64_tr_b8 v[12:13], v23
    ds_read_b64_tr_b8 v[14:15], v24

    ds_read_b64_tr_b8 v[16:17], v20 offset:64
    ds_read_b64_tr_b8 v[18:19], v20 offset:1024
    ds_read_b64_tr_b8 v[20:21], v20 offset:4160
    ds_read_b64_tr_b8 v[22:23], v20 offset:5120

    ds_read_b64_tr_b8 v[24:25], v25
    ds_read_b64_tr_b8 v[26:27], v26
    ds_read_b64_tr_b8 v[28:29], v27
    ds_read_b64_tr_b8 v[30:31], v28
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
.amdhsa_kernel _fwd_fp8_v_tr8_addr_map_debug
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
  - .name: _fwd_fp8_v_tr8_addr_map_debug
    .symbol: _fwd_fp8_v_tr8_addr_map_debug.kd
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

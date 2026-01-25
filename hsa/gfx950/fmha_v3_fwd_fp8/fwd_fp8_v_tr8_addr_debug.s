// Debug kernel: dump TR8 LDS read addresses (no data reads)
.text
.p2align 8
.globl _fwd_fp8_v_tr8_addr_debug
.type _fwd_fp8_v_tr8_addr_debug, @function

_fwd_fp8_v_tr8_addr_debug:
    s_mov_b64 exec, -1
    // Load kernel args (scaffold-compatible)
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr (unused)
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr (unused)
    s_load_dwordx2 s[16:17], s[0:1], 24   // V_ptr (unused)
    s_load_dword s20, s[0:1], 32          // num_k_tiles (unused)
    s_load_dword s21, s[0:1], 36          // stride_qh (unused)
    s_load_dword s22, s[0:1], 40          // stride_kh (unused)
    s_load_dword s23, s[0:1], 44          // stride_vh (unused)
    s_load_dword s24, s[0:1], 48          // stride_oh (unused)
    s_waitcnt lgkmcnt(0)

    // Buffer descriptor for O_ptr (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000

    v_mov_b32_e32 v60, v0                 // tid

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

    // Pack 16 addresses into v0..v15
    v_add_u32_e32 v0, 0, v20
    v_add_u32_e32 v1, 1088, v20
    v_add_u32_e32 v2, 4096, v20
    v_add_u32_e32 v3, 5184, v20
    v_mov_b32_e32 v4, v21
    v_mov_b32_e32 v5, v22
    v_mov_b32_e32 v6, v23
    v_mov_b32_e32 v7, v24
    v_add_u32_e32 v8, 64, v20
    v_add_u32_e32 v9, 1024, v20
    v_add_u32_e32 v10, 4160, v20
    v_add_u32_e32 v11, 5120, v20
    v_mov_b32_e32 v12, v25
    v_mov_b32_e32 v13, v26
    v_mov_b32_e32 v14, v27
    v_mov_b32_e32 v15, v28

    // Store 16 dwords per thread
    v_lshlrev_b32_e32 v30, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[0:3], v30, s[4:7], 0 offen
    v_add_u32_e32 v31, 16, v30
    buffer_store_dwordx4 v[4:7], v31, s[4:7], 0 offen
    v_add_u32_e32 v31, 32, v30
    buffer_store_dwordx4 v[8:11], v31, s[4:7], 0 offen
    v_add_u32_e32 v31, 48, v30
    buffer_store_dwordx4 v[12:15], v31, s[4:7], 0 offen

    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_v_tr8_addr_debug
    .amdhsa_group_segment_fixed_size 0
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
  - .name: _fwd_fp8_v_tr8_addr_debug
    .symbol: _fwd_fp8_v_tr8_addr_debug.kd
    .kernarg_segment_size: 52
    .group_segment_fixed_size: 0
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

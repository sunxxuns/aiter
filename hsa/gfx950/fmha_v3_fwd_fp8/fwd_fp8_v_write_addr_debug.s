// Debug kernel: dump LDS write address (bitop3:0x78)
.text
.p2align 8
.globl _fwd_fp8_v_write_addr_debug
.type _fwd_fp8_v_write_addr_debug, @function

_fwd_fp8_v_write_addr_debug:
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

    // Buffer descriptor for O_ptr
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000

    v_mov_b32_e32 v60, v0                 // tid
    v_lshlrev_b32_e32 v4, 4, v60          // tid * 16 bytes
    s_movk_i32 s26, 0x70
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x78

    // Store address (1 dword per thread)
    v_lshlrev_b32_e32 v30, 2, v60         // tid * 4 bytes
    buffer_store_dword v4, v30, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_v_write_addr_debug
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 52
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 4
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_v_write_addr_debug
    .symbol: _fwd_fp8_v_write_addr_debug.kd
    .kernarg_segment_size: 52
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 32
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

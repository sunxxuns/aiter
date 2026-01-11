// Test buffer_load with scalar offset
// Loads from K at two different offsets and sums them

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter12debug_bufloadE
.p2align 8
.type _ZN5aiter12debug_bufloadE,@function

_ZN5aiter12debug_bufloadE:
    s_mov_b64 exec, -1
    
    // Load kernel args
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // K input ptr
    s_load_dword s16, s[0:1], 0x10         // offset2 (bytes)
    
    v_and_b32_e32 v0, 63, v0               // tid = v0 % 64
    s_waitcnt lgkmcnt(0)
    
    // Setup K buffer descriptor (same pattern as working kernel)
    // s[8:9] = base addr (already loaded)
    // s10 = size
    // s11 = flags
    s_mov_b32 s10, 0x100000                 // 1MB size
    s_mov_b32 s11, 0x20000                  // offen mode
    
    // Each thread calculates its byte offset
    v_lshlrev_b32_e32 v1, 2, v0             // v1 = tid * 4 bytes
    
    // Load from K at offset 0
    s_mov_b32 s12, 0                        // scalar offset = 0
    buffer_load_dword v2, v1, s[8:11], s12 offen
    s_waitcnt vmcnt(0)
    
    // Load from K at offset2 (second tile)
    buffer_load_dword v3, v1, s[8:11], s16 offen
    s_waitcnt vmcnt(0)
    
    // Sum
    v_add_f32_e32 v4, v2, v3
    
    // Store to O using flat_store (known working)
    v_mov_b32_e32 v6, s4
    v_mov_b32_e32 v7, s5
    v_add_co_u32_e32 v6, vcc, v1, v6
    v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
    flat_store_dword v[6:7], v4
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter12debug_bufloadE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 12
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 12
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter12debug_bufloadE
    .symbol: _ZN5aiter12debug_bufloadE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 12
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: offset2, .size: 4, .offset: 16, .value_kind: by_value}
...
.end_amdgpu_metadata

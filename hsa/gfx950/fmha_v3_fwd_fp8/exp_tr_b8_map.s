// TR_B8 mapping experiment
// Goal: For each lane, determine exactly which LDS bytes it reads

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13exp_tr_b8_mapE
.p2align 8
.type _ZN5aiter13exp_tr_b8_mapE,@function

_ZN5aiter13exp_tr_b8_mapE:
    s_mov_b64 exec, -1
    
    s_load_dwordx2 s[4:5], s[0:1], 0     // output ptr
    s_load_dword s6, s[0:1], 8           // base_addr
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0   // lane_id
    
    // Fill LDS[0:1023] with address pattern: LDS[i] = i % 256
    // Use all 64 lanes to write 16 bytes each = 1024 bytes
    v_lshlrev_b32_e32 v1, 4, v0          // lane * 16
    
    // Build pattern: bytes [lane*16, lane*16+1, ..., lane*16+15] mod 256
    v_lshlrev_b32_e32 v2, 4, v0          // base = lane * 16
    
    // Byte 0-3
    v_and_b32_e32 v3, 0xff, v2
    v_add_u32_e32 v4, 1, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 8, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 2, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 16, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 3, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 24, v4
    v_or_b32_e32 v10, v3, v4
    
    // Byte 4-7
    v_add_u32_e32 v2, 4, v2
    v_and_b32_e32 v3, 0xff, v2
    v_add_u32_e32 v4, 1, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 8, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 2, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 16, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 3, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 24, v4
    v_or_b32_e32 v11, v3, v4
    
    // Byte 8-11
    v_add_u32_e32 v2, 4, v2
    v_and_b32_e32 v3, 0xff, v2
    v_add_u32_e32 v4, 1, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 8, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 2, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 16, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 3, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 24, v4
    v_or_b32_e32 v12, v3, v4
    
    // Byte 12-15
    v_add_u32_e32 v2, 4, v2
    v_and_b32_e32 v3, 0xff, v2
    v_add_u32_e32 v4, 1, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 8, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 2, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 16, v4
    v_or_b32_e32 v3, v3, v4
    v_add_u32_e32 v4, 3, v2
    v_and_b32_e32 v4, 0xff, v4
    v_lshlrev_b32_e32 v4, 24, v4
    v_or_b32_e32 v13, v3, v4
    
    ds_write_b128 v1, v[10:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Now read with TR_B8
    // Each lane uses same base_addr, TR_B8 distributes data across lanes
    v_mov_b32_e32 v20, s6
    ds_read_b64_tr_b8 v[30:31], v20
    s_waitcnt lgkmcnt(0)
    
    // Output: lane writes its result
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 3, v60
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx2 v[40:41], v[30:31]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13exp_tr_b8_mapE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter13exp_tr_b8_mapE
    .symbol: _ZN5aiter13exp_tr_b8_mapE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: base_addr, .size: 4, .offset: 8, .value_kind: by_value}
...
.end_amdgpu_metadata

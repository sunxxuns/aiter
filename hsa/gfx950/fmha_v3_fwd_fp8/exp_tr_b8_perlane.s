// TR_B8 with per-lane addresses
// Each lane uses different base_addr = lane * 8

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter17exp_tr_b8_perlaneE
.p2align 8
.type _ZN5aiter17exp_tr_b8_perlaneE,@function

_ZN5aiter17exp_tr_b8_perlaneE:
    s_mov_b64 exec, -1
    
    s_load_dwordx2 s[4:5], s[0:1], 0     // output ptr
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0   // lane_id
    
    // Fill LDS[0:1023] with LDS[i] = i % 256
    v_lshlrev_b32_e32 v1, 4, v0
    v_lshlrev_b32_e32 v2, 4, v0
    
    // Build 16 bytes per lane
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
    
    // TR_B8 with per-lane address: base = lane * 8
    v_lshlrev_b32_e32 v20, 3, v60        // lane * 8
    ds_read_b64_tr_b8 v[30:31], v20
    s_waitcnt lgkmcnt(0)
    
    // Output
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
.amdhsa_kernel _ZN5aiter17exp_tr_b8_perlaneE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
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
  - .name: _ZN5aiter17exp_tr_b8_perlaneE
    .symbol: _ZN5aiter17exp_tr_b8_perlaneE.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

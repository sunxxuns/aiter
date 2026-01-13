// Minimal 128Q test - just load Q and output wave_id

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16test_128q_qloadE
.p2align 8
.type _ZN5aiter16test_128q_qloadE,@function

_ZN5aiter16test_128q_qloadE:
    s_mov_b64 exec, -1
    
    // Args: O, Q
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    
    // Thread setup
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id (0-3)
    v_and_b32_e32 v60, 63, v0             // lane_id (0-63)
    v_readfirstlane_b32 s28, v61          // s28 = wave_id
    
    s_waitcnt lgkmcnt(0)
    
    // Load Q to LDS (4 iterations for 128 rows)
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Iteration 0
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    // Iteration 1
    v_add_u32_e32 v1, 4096, v1
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    // Iteration 2
    v_add_u32_e32 v1, 4096, v1
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    // Iteration 3
    v_add_u32_e32 v1, 4096, v1
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Output: each wave outputs its wave_id
    s_mul_i32 s40, s28, 16384             // wave_id * 16384
    
    // Per-lane offset
    v_lshlrev_b32_e32 v1, 6, v60          // lane * 64 bytes
    v_add_u32_e32 v1, s40, v1
    
    v_mov_b32_e32 v10, s4
    v_mov_b32_e32 v11, s5
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Write wave_id to output
    v_cvt_f32_u32_e32 v80, v61            // wave_id as float
    v_mov_b32_e32 v81, v80
    v_mov_b32_e32 v82, v80
    v_mov_b32_e32 v83, v80
    
    flat_store_dwordx4 v[10:11], v[80:83]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16test_128q_qloadE
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 96
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 96
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter16test_128q_qloadE
    .symbol: _ZN5aiter16test_128q_qloadE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 96
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

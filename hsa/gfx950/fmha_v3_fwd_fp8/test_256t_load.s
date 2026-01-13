// Minimal 256T test - just cooperative load Q to LDS and output

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter14test_256t_loadE
.p2align 8
.type _ZN5aiter14test_256t_loadE,@function

_ZN5aiter14test_256t_loadE:
    s_mov_b64 exec, -1
    
    // Args: output, Q_ptr
    s_load_dwordx2 s[4:5], s[0:1], 0      // output
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr
    s_waitcnt lgkmcnt(0)
    
    // Thread setup
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id (0-3)
    v_and_b32_e32 v60, 63, v0             // lane_id (0-63)
    
    // 256 threads load 4KB Q
    // Each thread loads 16 bytes at offset tid*16
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16
    
    // Global addr
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    // Write to LDS
    ds_write_b128 v1, v[20:23]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Read back and output (wave 0 only)
    v_cmp_eq_u32_e64 vcc, v61, 0
    s_and_saveexec_b64 s[12:13], vcc
    
    // Read from LDS
    v_lshlrev_b32_e32 v1, 4, v60          // lane * 16
    ds_read_b128 v[30:33], v1
    s_waitcnt lgkmcnt(0)
    
    // Output
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 4, v60
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx4 v[40:41], v[30:33]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter14test_256t_loadE
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
  - .name: _ZN5aiter14test_256t_loadE
    .symbol: _ZN5aiter14test_256t_loadE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 256
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

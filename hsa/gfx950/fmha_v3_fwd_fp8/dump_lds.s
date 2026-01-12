
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter8dump_ldsE
.p2align 8

_ZN5aiter8dump_ldsE:
    s_mov_b64 exec, -1
    
    // Args: O, Q
    s_load_dwordx2 s[4:5], s[0:1], 0x00
    s_load_dwordx2 s[8:9], s[0:1], 0x08
    s_waitcnt lgkmcnt(0)
    
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // Load Q to LDS (like 64T kernel - 4 passes)
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16
    
    s_mov_b32 m0, 0
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 1024
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 2048
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 3072
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // Read back from LDS and output (just first 256 bytes as F32)
    v_lshlrev_b32_e32 v2, 2, v0           // tid * 4 (output offset)
    ds_read_b32 v3, v2                    // Read 4 bytes from LDS
    s_waitcnt lgkmcnt(0)
    
    // Output
    buffer_store_dword v3, v2, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter8dump_ldsE
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 8
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter8dump_ldsE
    .symbol: _ZN5aiter8dump_ldsE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 8192
    .max_flat_workgroup_size: 64
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer}
...
.end_amdgpu_metadata

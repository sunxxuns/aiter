
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl _ZN5aiter7test_q2E
.p2align 8

_ZN5aiter7test_q2E:
    s_mov_b64 exec, -1
    
    // Decompose thread ID
    v_lshrrev_b32_e32 v1, 6, v0           // wave_id
    v_and_b32_e32 v0, 63, v0              // lane_id
    
    s_load_dwordx2 s[4:5], s[0:1], 0x00   // O
    s_load_dwordx2 s[8:9], s[0:1], 0x08   // Q
    s_waitcnt lgkmcnt(0)
    
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // Load Q to LDS (256 threads × 16 bytes)
    v_lshlrev_b32_e32 v2, 6, v1           // wave*64
    v_add_u32_e32 v2, v0, v2              // thread_id
    v_lshlrev_b32_e32 v2, 4, v2           // thread_id * 16
    
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v2, s[8:11], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // Read first 4 bytes from LDS (thread 0 only)
    v_cmp_eq_u32_e32 vcc, 0, v0
    v_cmp_eq_u32_e64 s[20:21], 0, v1
    s_and_b64 vcc, vcc, s[20:21]
    s_and_saveexec_b64 s[20:21], vcc
    s_cbranch_execz SKIP_OUTPUT
    
    // Read 4 bytes from LDS offset 0 (should be first 4 bytes of Q)
    v_mov_b32_e32 v3, 0
    ds_read_b32 v4, v3
    s_waitcnt lgkmcnt(0)
    
    // Output
    v_mov_b32_e32 v5, 0
    buffer_store_dword v4, v5, s[4:7], 0 offen

SKIP_OUTPUT:
    s_mov_b64 exec, -1
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter7test_q2E
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
  - .name: _ZN5aiter7test_q2E
    .symbol: _ZN5aiter7test_q2E.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 8192
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0}
      - {.name: Q, .size: 8, .offset: 8}
...
.end_amdgpu_metadata

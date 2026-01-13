// 256T QK-only test (no softmax, no PV)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter12test_256t_qkE
.p2align 8
.type _ZN5aiter12test_256t_qkE,@function

_ZN5aiter12test_256t_qkE:
    s_mov_b64 exec, -1
    
    // Args: output, K_ptr, Q_ptr
    s_load_dwordx2 s[4:5], s[0:1], 0      // output
    s_load_dwordx2 s[8:9], s[0:1], 8      // K
    s_load_dwordx2 s[12:13], s[0:1], 16   // Q
    s_waitcnt lgkmcnt(0)
    
    // Thread setup
    v_mov_b32_e32 v59, v0                 // save tid
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id
    v_and_b32_e32 v60, 63, v0             // lane_id
    
    // ========================================================================
    // COOPERATIVE Q LOAD (256 threads load 32×128 = 4KB)
    // ========================================================================
    
    v_lshlrev_b32_e32 v1, 4, v59          // tid * 16
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COOPERATIVE K LOAD to LDS at offset 4096
    // ========================================================================
    
    v_lshlrev_b32_e32 v1, 4, v59
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v1, 4096, v1
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // MFMA ROW MAPPING
    // ========================================================================
    
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v62, v3, v4             // mfma_row
    
    v_mov_b32_e32 v63, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e32 v63, v63, v4, vcc   // k_base
    
    // LDS addresses
    v_lshlrev_b32_e32 v70, 7, v62         // Q: mfma_row * 128 + k_base
    v_add_u32_e32 v70, v70, v63
    
    v_lshlrev_b32_e32 v71, 7, v62         // K: 4096 + mfma_row * 128 + k_base
    v_add_u32_e32 v71, v71, v63
    v_add_u32_e32 v71, 4096, v71
    
    // Clear accumulators
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0
    v_mov_b32_e32 v13, 0
    v_mov_b32_e32 v14, 0
    v_mov_b32_e32 v15, 0
    
    // 8 MFMAs for HD=128
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v71
        ds_read_b64 v[30:31], v72
        v_add_u32_e32 v73, \k_off, v70
        ds_read_b64 v[32:33], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    
    s_nop 15
    
    // ========================================================================
    // OUTPUT (wave 0 only)
    // ========================================================================
    
    v_cmp_eq_u32_e64 vcc, v61, 0
    s_and_saveexec_b64 s[16:17], vcc
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx4 v[40:41], v[0:3]
    v_add_co_u32_e32 v42, vcc, 16, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[4:7]
    v_add_co_u32_e32 v42, vcc, 32, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[8:11]
    v_add_co_u32_e32 v42, vcc, 48, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[12:15]
    
    s_mov_b64 exec, s[16:17]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter12test_256t_qkE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter12test_256t_qkE
    .symbol: _ZN5aiter12test_256t_qkE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 80
    .max_flat_workgroup_size: 256
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

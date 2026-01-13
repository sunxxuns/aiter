// FP8 QK MFMA with stride-132 - FAST version
// Uses row*132 = row*128 + row*4 = (row<<7) + (row<<2) instead of multiply

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set LDS_STRIDE, 132
.set HD, 128
.set Q_LDS_BASE, 0
.set K_LDS_BASE, 4352

.text
.globl _ZN5aiter20qk_fp8_stride132_fastE
.p2align 8
.type _ZN5aiter20qk_fp8_stride132_fastE,@function

_ZN5aiter20qk_fp8_stride132_fastE:
    s_mov_b64 exec, -1
    
    s_load_dwordx4 s[4:7], s[0:1], 0
    s_load_dwordx2 s[10:11], s[0:1], 16
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0
    
    // ========================================================================
    // LOAD Q - use fast address: row*132 = (row<<7) + (row<<2)
    // ========================================================================
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    v_lshlrev_b32_e32 v1, 7, v0         // lane * 128 for global addr
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS addr: lane*132 = (lane<<7) + (lane<<2)
    v_lshlrev_b32_e32 v5, 7, v0         // lane * 128
    v_lshlrev_b32_e32 v6, 2, v0         // lane * 4
    v_add_u32_e32 v5, v5, v6            // lane * 132
    
    flat_load_dwordx4 v[20:23], v[10:11] offset:0
    flat_load_dwordx4 v[24:27], v[10:11] offset:16
    flat_load_dwordx4 v[28:31], v[10:11] offset:32
    flat_load_dwordx4 v[32:35], v[10:11] offset:48
    flat_load_dwordx4 v[36:39], v[10:11] offset:64
    flat_load_dwordx4 v[40:43], v[10:11] offset:80
    flat_load_dwordx4 v[44:47], v[10:11] offset:96
    flat_load_dwordx4 v[48:51], v[10:11] offset:112
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v6, 0, v5
    ds_write_b128 v6, v[20:23]
    v_add_u32_e32 v6, 16, v5
    ds_write_b128 v6, v[24:27]
    v_add_u32_e32 v6, 32, v5
    ds_write_b128 v6, v[28:31]
    v_add_u32_e32 v6, 48, v5
    ds_write_b128 v6, v[32:35]
    v_add_u32_e32 v6, 64, v5
    ds_write_b128 v6, v[36:39]
    v_add_u32_e32 v6, 80, v5
    ds_write_b128 v6, v[40:43]
    v_add_u32_e32 v6, 96, v5
    ds_write_b128 v6, v[44:47]
    v_add_u32_e32 v6, 112, v5
    ds_write_b128 v6, v[48:51]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K
    // ========================================================================
    s_mov_b64 exec, -1
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    v_lshlrev_b32_e32 v1, 7, v0
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    v_lshlrev_b32_e32 v5, 7, v0
    v_lshlrev_b32_e32 v6, 2, v0
    v_add_u32_e32 v5, v5, v6
    v_add_u32_e32 v5, K_LDS_BASE, v5
    
    flat_load_dwordx4 v[20:23], v[10:11] offset:0
    flat_load_dwordx4 v[24:27], v[10:11] offset:16
    flat_load_dwordx4 v[28:31], v[10:11] offset:32
    flat_load_dwordx4 v[32:35], v[10:11] offset:48
    flat_load_dwordx4 v[36:39], v[10:11] offset:64
    flat_load_dwordx4 v[40:43], v[10:11] offset:80
    flat_load_dwordx4 v[44:47], v[10:11] offset:96
    flat_load_dwordx4 v[48:51], v[10:11] offset:112
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v6, 0, v5
    ds_write_b128 v6, v[20:23]
    v_add_u32_e32 v6, 16, v5
    ds_write_b128 v6, v[24:27]
    v_add_u32_e32 v6, 32, v5
    ds_write_b128 v6, v[28:31]
    v_add_u32_e32 v6, 48, v5
    ds_write_b128 v6, v[32:35]
    v_add_u32_e32 v6, 64, v5
    ds_write_b128 v6, v[36:39]
    v_add_u32_e32 v6, 80, v5
    ds_write_b128 v6, v[40:43]
    v_add_u32_e32 v6, 96, v5
    ds_write_b128 v6, v[44:47]
    v_add_u32_e32 v6, 112, v5
    ds_write_b128 v6, v[48:51]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_mov_b64 exec, -1
    
    // ========================================================================
    // MFMA row mapping + fast address calculation
    // ========================================================================
    v_and_b32_e32 v1, 3, v0
    v_lshrrev_b32_e32 v2, 3, v0
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    
    v_lshrrev_b32_e32 v4, 2, v0
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v61, v3, v4            // full_row
    
    v_mov_b32_e32 v62, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v0, 32
    v_cndmask_b32_e32 v62, v62, v4, vcc  // k_base
    
    // Fast: row*132 = (row<<7) + (row<<2)
    v_lshlrev_b32_e32 v63, 7, v61        // row * 128
    v_lshlrev_b32_e32 v64, 2, v61        // row * 4
    v_add_u32_e32 v63, v63, v64          // row * 132
    v_add_u32_e32 v63, v63, v62          // + k_base
    
    v_add_u32_e32 v70, K_LDS_BASE, v63
    v_mov_b32_e32 v71, v63
    
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
    
    // 8 MFMAs
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v70
        ds_read_b64 v[30:31], v72
        v_add_u32_e32 v73, \k_off, v71
        ds_read_b64 v[32:33], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    
    s_nop 15
    
    // Output
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
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter20qk_fp8_stride132_fastE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter20qk_fp8_stride132_fastE
    .symbol: _ZN5aiter20qk_fp8_stride132_fastE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 80
    .max_flat_workgroup_size: 64
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

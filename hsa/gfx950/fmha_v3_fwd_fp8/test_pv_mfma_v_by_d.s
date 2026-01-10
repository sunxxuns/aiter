// Test: V varies by D position, P=1
// V[K, D] = D/32, P[Q, K] = 1
// After V×P MFMA: O[Q, D] = sum over K of V^T[D, K] * P^T[K, Q]
//                        = sum over K of (D/32) * 1
//                        = 16 * D/32 = D/2
// So O[Q, D] should be D/2 (for 16 K values)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter18test_pv_v_by_dE
.p2align 8
.type _ZN5aiter18test_pv_v_by_dE,@function

_ZN5aiter18test_pv_v_by_dE:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_out
    
    v_and_b32_e32 v0, 63, v0               // lane_id (0-63)
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize output accumulators
    v_mov_b32_e32 v32, 0
    v_mov_b32_e32 v33, 0
    v_mov_b32_e32 v34, 0
    v_mov_b32_e32 v35, 0
    v_mov_b32_e32 v36, 0
    v_mov_b32_e32 v37, 0
    v_mov_b32_e32 v38, 0
    v_mov_b32_e32 v39, 0
    v_mov_b32_e32 v40, 0
    v_mov_b32_e32 v41, 0
    v_mov_b32_e32 v42, 0
    v_mov_b32_e32 v43, 0
    v_mov_b32_e32 v44, 0
    v_mov_b32_e32 v45, 0
    v_mov_b32_e32 v46, 0
    v_mov_b32_e32 v47, 0
    
    // ========================================================================
    // Generate V values: V^T[D, K] = D/32
    // A operand: Thread t provides V^T[D=t%32, K_range]
    // ========================================================================
    
    v_and_b32_e32 v1, 31, v0              // D = lane_id % 32
    v_cvt_f32_u32_e32 v2, v1              // D as float
    v_mul_f32_e32 v2, 0x3d000000, v2      // D * (1/32) = D/32
    
    // Convert V float to FP8 and replicate to 8 values
    v_mov_b32_e32 v4, 0
    v_cvt_pk_fp8_f32 v4, v2, v2           // 2 identical V values (D/32)
    v_lshlrev_b32_e32 v5, 16, v4
    v_or_b32_e32 v4, v4, v5               // 4 identical V values
    v_mov_b32_e32 v5, v4                  // v5 also has 4 V values
    v_accvgpr_write_b32 a0, v4
    v_accvgpr_write_b32 a1, v5
    
    // ========================================================================
    // Generate P values: P^T[K, Q] = 1
    // B operand: Thread t provides P^T[K_range, Q=t%32] = 1
    // ========================================================================
    
    v_mov_b32_e32 v6, 0x3f800000          // P = 1.0
    v_mov_b32_e32 v7, 0
    v_cvt_pk_fp8_f32 v7, v6, v6           // 2 P values = 1.0
    v_lshlrev_b32_e32 v8, 16, v7
    v_or_b32_e32 v7, v7, v8               // 4 P values
    v_mov_b32_e32 v8, v7
    v_mov_b32_e32 v64, v7
    v_mov_b32_e32 v65, v8
    
    s_nop 7
    
    // V×P MFMA
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 7
    
    // Store output with transposed pattern
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0              // Q = t%32
    v_lshrrev_b32_e32 v4, 5, v0           // D_group = t/32
    v_lshlrev_b32_e32 v3, 9, v3           // Q * 512
    v_lshlrev_b32_e32 v4, 6, v4           // D_group * 64
    v_add_u32_e32 v3, v3, v4
    
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_store_dwordx4 v[10:11], v[32:35]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[36:39]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[40:43]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[44:47]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter18test_pv_v_by_dE, .-_ZN5aiter18test_pv_v_by_dE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter18test_pv_v_by_dE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 68
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter18test_pv_v_by_dE
    .symbol: _ZN5aiter18test_pv_v_by_dE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 80
    .agpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - .name: ptr_out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

// Test: P operand layout from QK MFMA output
//
// After QK MFMA, thread t has P[Q_base:Q_base+16, K=t%32]
// where Q_base = (t/32)*16
//
// For P×V MFMA A operand, we need P[M=f(t), K_range]
// where M is the Q position and K_range spans K values.
//
// This test simulates the P layout issue:
// - Generate P values as if from QK MFMA (16 Q values at single K)
// - Use them as A operand for P×V
// - See if the output is correct
//
// If there's a layout mismatch, the output will be wrong.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16test_p_layout_gpuE
.p2align 8
.type _ZN5aiter16test_p_layout_gpuE,@function

_ZN5aiter16test_p_layout_gpuE:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize output
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
    // Simulate QK MFMA output layout:
    // Thread t has P[Q_base:Q_base+16, K=t%32]
    // where Q_base = (t/32)*16
    //
    // For simplicity, P[q, k] = 1 for k < 8, 0 for k >= 8
    // So threads 0-7 and 32-39 have P=1, others have P=0
    // Total P sum per Q row = 8
    //
    // If P×V is computed correctly:
    // O[q,d] = sum_k(P[q,k] * V[k,d])
    //        = sum over k=0..7 of (1 * V[k,d]) 
    //        = sum over k=0..7 of V[k,d]
    //
    // With V[k,d] = 1: O[q,d] = 8
    // With V[k,d] = d: O[q,d] = 8*d
    // ========================================================================
    
    // Generate P values based on K position
    // K = t % 32
    // P = 1 if K < 8, else 0
    v_and_b32_e32 v1, 31, v0              // K = lane_id % 32
    v_cmp_lt_u32_e64 s[0:1], v1, 8        // K < 8?
    v_cndmask_b32_e64 v2, 0, 0x3f800000, s[0:1]  // P = 1.0 if K<8, else 0
    
    // Pack P values (8 identical values)
    v_mov_b32_e32 v4, 0
    v_cvt_pk_fp8_f32 v4, v2, v2
    v_lshlrev_b32_e32 v5, 16, v4
    v_or_b32_e32 v4, v4, v5
    v_mov_b32_e32 v5, v4
    
    v_accvgpr_write_b32 a0, v4
    v_accvgpr_write_b32 a1, v5
    
    // V operand: V[k,d] = d (D index)
    // This tests if D values are preserved through P×V
    v_and_b32_e32 v1, 31, v0              // d = lane_id % 32
    v_cvt_f32_u32_e32 v6, v1
    
    v_mov_b32_e32 v64, 0
    v_cvt_pk_fp8_f32 v64, v6, v6
    v_lshlrev_b32_e32 v65, 16, v64
    v_or_b32_e32 v64, v64, v65
    v_mov_b32_e32 v65, v64
    
    s_nop 7
    
    // P×V MFMA
    // With P=1 for K<8, V[k,d]=d:
    // O[q,d] = sum over K=0..7 of (1 * d) = 8 * d
    // Since we're only using K=0..15 in one MFMA (16 K values),
    // and P=1 for K<8, we get: sum over K=0..7 of d = 8*d
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 7
    
    // Store
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_lshlrev_b32_e32 v3, 6, v0
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

.size _ZN5aiter16test_p_layout_gpuE, .-_ZN5aiter16test_p_layout_gpuE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16test_p_layout_gpuE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
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
  - .name: _ZN5aiter16test_p_layout_gpuE
    .symbol: _ZN5aiter16test_p_layout_gpuE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
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

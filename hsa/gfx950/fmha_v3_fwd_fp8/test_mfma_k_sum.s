// Test: MFMA with A[m,k]=1, B[k,n]=k -> C[m,n]=sum(k)=120
// This tests K reduction - all outputs should be 120.0
// Uses K values 0-15 which FP8 can represent exactly.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16test_mfma_k_sumE
.p2align 8
.type _ZN5aiter16test_mfma_k_sumE,@function

_ZN5aiter16test_mfma_k_sumE:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00
    
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
    // A operand: A[m,k] = 1.0
    // All threads provide 8 ones
    // ========================================================================
    
    v_mov_b32_e32 v2, 0x3f800000          // 1.0
    v_mov_b32_e32 v4, 0
    v_cvt_pk_fp8_f32 v4, v2, v2           // 2 ones
    v_lshlrev_b32_e32 v5, 16, v4
    v_or_b32_e32 v4, v4, v5               // 4 ones
    v_mov_b32_e32 v5, v4                  // 8 ones
    
    v_accvgpr_write_b32 a0, v4
    v_accvgpr_write_b32 a1, v5
    
    // ========================================================================
    // B operand: B[k,n] = k
    // For MFMA B operand, thread provides values for specific N column
    // The K values need to reflect the K dimension position
    //
    // MFMA 32x32x16: B is 16K x 32N
    // Thread t provides B[K_range, N=t%32]
    // K_range = (t/32)*8 : (t/32)*8+8
    // So thread 0-31 provide K=0..7, threads 32-63 provide K=8..15
    // ========================================================================
    
    v_lshrrev_b32_e32 v1, 5, v0           // k_group = lane_id / 32 (0 or 1)
    v_lshlrev_b32_e32 v1, 3, v1           // k_base = k_group * 8 (0 or 8)
    
    // Generate K values: k_base, k_base+1, ..., k_base+7
    v_cvt_f32_u32_e32 v6, v1              // k_base as float
    
    // k_base+0, k_base+1
    v_add_f32_e32 v7, 1.0, v6             // k_base + 1
    v_mov_b32_e32 v64, 0
    v_cvt_pk_fp8_f32 v64, v6, v7          // [k_base, k_base+1]
    
    // k_base+2, k_base+3
    v_add_f32_e32 v8, 2.0, v6
    v_add_f32_e32 v9, 3.0, v6
    v_mov_b32_e32 v65, 0
    v_cvt_pk_fp8_f32 v65, v8, v9          // [k_base+2, k_base+3]
    
    // Combine into first dword
    v_lshlrev_b32_e32 v65, 16, v65
    v_or_b32_e32 v64, v64, v65            // [k_base..k_base+3]
    
    // k_base+4, k_base+5
    v_add_f32_e32 v8, 4.0, v6
    v_add_f32_e32 v9, 5.0, v6
    v_mov_b32_e32 v66, 0
    v_cvt_pk_fp8_f32 v66, v8, v9
    
    // k_base+6, k_base+7
    v_add_f32_e32 v8, 6.0, v6
    v_add_f32_e32 v9, 7.0, v6
    v_mov_b32_e32 v67, 0
    v_cvt_pk_fp8_f32 v67, v8, v9
    
    // Combine into second dword
    v_lshlrev_b32_e32 v67, 16, v67
    v_or_b32_e32 v65, v66, v67            // [k_base+4..k_base+7]
    
    s_nop 7
    
    // MFMA: C = A × B = 1 × K = K
    // sum(k for k=0..15) = 0+1+2+...+15 = 120
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 7
    
    // Store output
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

.size _ZN5aiter16test_mfma_k_sumE, .-_ZN5aiter16test_mfma_k_sumE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16test_mfma_k_sumE
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
  - .name: _ZN5aiter16test_mfma_k_sumE
    .symbol: _ZN5aiter16test_mfma_k_sumE.kd
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

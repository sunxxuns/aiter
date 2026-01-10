// Test: MFMA with A[m,k]=m, B[k,n]=1 -> C[m,n]=m*K
// This reveals which M rows each thread owns in the output.
// Expected output: C[m,n] = m * 16 for each M position

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16test_mfma_m_idxE
.p2align 8
.type _ZN5aiter16test_mfma_m_idxE,@function

_ZN5aiter16test_mfma_m_idxE:
    // Args: ptr_out (output buffer)
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00
    
    v_and_b32_e32 v0, 63, v0               // lane_id (0-63)
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize output accumulators to 0
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
    // A operand: A[m,k] = m (M row index)
    // For FP8 MFMA 32x32x16:
    // - A is 32M x 16K
    // - Each thread provides 8 FP8 values for specific M position
    // - Thread t provides A[M=f(t), K_range]
    // - For standard layout: M = t % 32, K_range depends on t/32
    //
    // Thread t needs to provide value "M" for its M position, replicated 8 times
    // ========================================================================
    
    // Compute M position for this thread
    // For MFMA A operand: M = lane_id % 32
    v_and_b32_e32 v1, 31, v0              // M = lane_id % 32
    v_cvt_f32_u32_e32 v2, v1              // M as float
    
    // Scale M to fit in FP8 range (M can be 0-31, multiply by some factor)
    // Using M directly (0-31) fits well in FP8
    
    // Pack 8 identical M values into 2 dwords for A operand
    v_mov_b32_e32 v4, 0
    v_cvt_pk_fp8_f32 v4, v2, v2           // 2 M values
    v_lshlrev_b32_e32 v5, 16, v4
    v_or_b32_e32 v4, v4, v5               // 4 M values
    v_mov_b32_e32 v5, v4                  // 8 M values total in v4,v5
    
    v_accvgpr_write_b32 a0, v4
    v_accvgpr_write_b32 a1, v5
    
    // ========================================================================
    // B operand: B[k,n] = 1
    // Each thread provides 8 FP8 values = 1.0
    // ========================================================================
    
    v_mov_b32_e32 v6, 0x3f800000          // 1.0
    v_mov_b32_e32 v7, 0
    v_cvt_pk_fp8_f32 v7, v6, v6           // 2 ones
    v_lshlrev_b32_e32 v8, 16, v7
    v_or_b32_e32 v7, v7, v8               // 4 ones
    v_mov_b32_e32 v8, v7                  // 8 ones
    v_mov_b32_e32 v64, v7
    v_mov_b32_e32 v65, v8
    
    s_nop 7
    
    // MFMA: C = A × B
    // With A[m,k]=m and B[k,n]=1: C[m,n] = sum_k(m*1) = m*K = m*16
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 7
    
    // ========================================================================
    // Store output
    // Thread t owns 16 output values at specific M positions and N=t%32
    // Store to output at layout that preserves thread->position mapping
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    // Store at tid * 64 bytes (16 floats per thread)
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

.size _ZN5aiter16test_mfma_m_idxE, .-_ZN5aiter16test_mfma_m_idxE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16test_mfma_m_idxE
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
  - .name: _ZN5aiter16test_mfma_m_idxE
    .symbol: _ZN5aiter16test_mfma_m_idxE.kd
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

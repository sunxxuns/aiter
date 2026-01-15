// FP8 QK + Softmax (Step 2)
// Computes P = softmax(Q @ K^T / sqrt(d))
// 256 threads (4 waves)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set SCALE, 0x3e028f5c              // log2(e) / sqrt(128) = 0.12754

.text
.globl _ZN5aiter15fwd_fp8_swizzleE
.p2align 8
.type _ZN5aiter15fwd_fp8_swizzleE,@function

_ZN5aiter15fwd_fp8_swizzleE:
    s_mov_b64 exec, -1
    
    // Args: output, Q_ptr, K_ptr
    s_load_dwordx2 s[4:5], s[0:1], 0      // output
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr
    s_waitcnt lgkmcnt(0)
    
    // Thread info
    v_and_b32_e32 v60, 63, v0             // lane_id (0-63)
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id
    
    // Scale constant
    s_mov_b32 s2, SCALE
    
    // ========================================================================
    // LOAD Q[32×128] TO LDS
    // ========================================================================
    
    v_lshlrev_b32_e32 v1, 4, v60          // lane * 16
    v_lshlrev_b32_e32 v2, 10, v61         // wave * 1024
    v_add_u32_e32 v4, v1, v2              // total offset
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    v_mov_b32_e32 v5, v4                  // LDS Q addr (base 0)
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v5, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K[32×128] TO LDS (offset 4096)
    // ========================================================================
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    v_add_u32_e32 v6, 4096, v4            // LDS K addr
    
    flat_load_dwordx4 v[24:27], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[24:27]
    
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
    v_add_u32_e32 v63, v3, v4             // mfma_row
    
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc      // k_half
    
    v_lshlrev_b32_e32 v10, 7, v63
    v_add_u32_e32 v10, v10, v64
    
    v_mov_b32_e32 v70, v10                // Q addr base
    v_add_u32_e32 v71, 4096, v10          // K addr base
    
    // ========================================================================
    // CLEAR ACCUMULATORS AND COMPUTE QK MFMA
    // ========================================================================
    
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
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
    
    // S is now in v[0:15] (16 F32 values per lane)
    // For a 32x32 tile, each of 64 lanes has 16 values
    // But we need to be careful: MFMA output layout is interleaved
    
    // ========================================================================
    // SOFTMAX: Find row max
    // v[0:15] contains 16 values per lane, need row-wise max
    // Row = col (lane % 32), values are interleaved
    // ========================================================================
    
    // First: local max of 16 values
    v_max_f32_e32 v20, v0, v1
    v_max_f32_e32 v20, v20, v2
    v_max_f32_e32 v20, v20, v3
    v_max_f32_e32 v20, v20, v4
    v_max_f32_e32 v20, v20, v5
    v_max_f32_e32 v20, v20, v6
    v_max_f32_e32 v20, v20, v7
    v_max_f32_e32 v20, v20, v8
    v_max_f32_e32 v20, v20, v9
    v_max_f32_e32 v20, v20, v10
    v_max_f32_e32 v20, v20, v11
    v_max_f32_e32 v20, v20, v12
    v_max_f32_e32 v20, v20, v13
    v_max_f32_e32 v20, v20, v14
    v_max_f32_e32 v20, v20, v15
    
    // Cross-lane max (lanes 0-31 swap with 32-63)
    v_mov_b32_e32 v21, v20
    s_nop 1
    v_permlane32_swap_b32_e32 v21, v20
    v_max_f32_e32 v20, v20, v21           // v20 = row_max
    
    // ========================================================================
    // SOFTMAX: exp((S - max) * scale)
    // ========================================================================
    
    v_mul_f32_e32 v21, s2, v20            // max * scale
    
    v_fma_f32 v0, v0, s2, -v21
    v_fma_f32 v1, v1, s2, -v21
    v_fma_f32 v2, v2, s2, -v21
    v_fma_f32 v3, v3, s2, -v21
    v_fma_f32 v4, v4, s2, -v21
    v_fma_f32 v5, v5, s2, -v21
    v_fma_f32 v6, v6, s2, -v21
    v_fma_f32 v7, v7, s2, -v21
    v_fma_f32 v8, v8, s2, -v21
    v_fma_f32 v9, v9, s2, -v21
    v_fma_f32 v10, v10, s2, -v21
    v_fma_f32 v11, v11, s2, -v21
    v_fma_f32 v12, v12, s2, -v21
    v_fma_f32 v13, v13, s2, -v21
    v_fma_f32 v14, v14, s2, -v21
    v_fma_f32 v15, v15, s2, -v21
    
    v_exp_f32_e32 v0, v0
    v_exp_f32_e32 v1, v1
    v_exp_f32_e32 v2, v2
    v_exp_f32_e32 v3, v3
    v_exp_f32_e32 v4, v4
    v_exp_f32_e32 v5, v5
    v_exp_f32_e32 v6, v6
    v_exp_f32_e32 v7, v7
    v_exp_f32_e32 v8, v8
    v_exp_f32_e32 v9, v9
    v_exp_f32_e32 v10, v10
    v_exp_f32_e32 v11, v11
    v_exp_f32_e32 v12, v12
    v_exp_f32_e32 v13, v13
    v_exp_f32_e32 v14, v14
    v_exp_f32_e32 v15, v15
    s_nop 7
    s_nop 7
    s_nop 7     // Wait for exp latency
    
    // ========================================================================
    // SOFTMAX: Sum and normalize
    // ========================================================================
    
    v_add_f32_e32 v22, v0, v1
    v_add_f32_e32 v22, v22, v2
    v_add_f32_e32 v22, v22, v3
    v_add_f32_e32 v22, v22, v4
    v_add_f32_e32 v22, v22, v5
    v_add_f32_e32 v22, v22, v6
    v_add_f32_e32 v22, v22, v7
    v_add_f32_e32 v22, v22, v8
    v_add_f32_e32 v22, v22, v9
    v_add_f32_e32 v22, v22, v10
    v_add_f32_e32 v22, v22, v11
    v_add_f32_e32 v22, v22, v12
    v_add_f32_e32 v22, v22, v13
    v_add_f32_e32 v22, v22, v14
    v_add_f32_e32 v22, v22, v15
    
    // Cross-lane sum
    v_mov_b32_e32 v23, v22
    s_nop 1
    v_permlane32_swap_b32_e32 v23, v22
    v_add_f32_e32 v22, v22, v23           // v22 = row_sum
    
    // Normalize: P = exp_val / sum
    v_rcp_f32_e32 v22, v22                // v22 = 1/sum
    s_nop 3
    
    v_mul_f32_e32 v0, v0, v22
    v_mul_f32_e32 v1, v1, v22
    v_mul_f32_e32 v2, v2, v22
    v_mul_f32_e32 v3, v3, v22
    v_mul_f32_e32 v4, v4, v22
    v_mul_f32_e32 v5, v5, v22
    v_mul_f32_e32 v6, v6, v22
    v_mul_f32_e32 v7, v7, v22
    v_mul_f32_e32 v8, v8, v22
    v_mul_f32_e32 v9, v9, v22
    v_mul_f32_e32 v10, v10, v22
    v_mul_f32_e32 v11, v11, v22
    v_mul_f32_e32 v12, v12, v22
    v_mul_f32_e32 v13, v13, v22
    v_mul_f32_e32 v14, v14, v22
    v_mul_f32_e32 v15, v15, v22
    
    // P is now in v[0:15], normalized softmax output
    
    // ========================================================================
    // STORE OUTPUT (P values)
    // ========================================================================
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60         // lane * 64
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
.amdhsa_kernel _ZN5aiter15fwd_fp8_swizzleE
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
  - .name: _ZN5aiter15fwd_fp8_swizzleE
    .symbol: _ZN5aiter15fwd_fp8_swizzleE.kd
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
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

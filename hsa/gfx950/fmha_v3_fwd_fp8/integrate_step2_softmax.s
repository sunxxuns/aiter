// Integration Step 2: QK MFMA + Softmax
// 1. Load Q[32,16] and K[32,16] from global memory
// 2. Compute S = Q @ K^T using MFMA
// 3. Store S to LDS
// 4. Compute P = softmax(S) for each row
// 5. Output P[32,32] to global memory
//
// For head_dim=16, single K-tile, seq_len=32

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set LDS_S, 0          // S[32,32] at offset 0, 4KB

.text
.globl _ZN5aiter20integrate_step2_softmaxE
.p2align 8
.type _ZN5aiter20integrate_step2_softmaxE,@function

_ZN5aiter20integrate_step2_softmaxE:
    // Args: 0=ptr_out, 8=ptr_Q, 16=ptr_K
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_out
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // ptr_K
    
    v_and_b32_e32 v0, 63, v0               // tid
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize accumulators
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
    // Load Q for A operand
    // ========================================================================
    v_and_b32_e32 v1, 31, v0              // q = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // d_group = tid / 32
    v_lshlrev_b32_e32 v2, 3, v2           // d_start = d_group * 8
    
    v_lshlrev_b32_e32 v3, 4, v1           // q * 16
    v_add_u32_e32 v3, v3, v2              // + d_start
    
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[20:21], v[10:11]
    
    // ========================================================================
    // Load K for B operand
    // ========================================================================
    v_and_b32_e32 v1, 31, v0              // k = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // d_group = tid / 32
    v_lshlrev_b32_e32 v2, 3, v2           // d_start = d_group * 8
    
    v_lshlrev_b32_e32 v3, 4, v1           // k * 16
    v_add_u32_e32 v3, v3, v2              // + d_start
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[64:65], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    
    s_nop 7
    
    // ========================================================================
    // QK MFMA: S = Q @ K^T
    // ========================================================================
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 15
    s_nop 7
    
    // ========================================================================
    // Store S to LDS
    // Same scatter pattern as global store
    // ========================================================================
    v_and_b32_e32 v3, 31, v0              // N = tid % 32
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4 bytes
    v_lshrrev_b32_e32 v4, 5, v0           // M_base_idx = tid / 32
    v_lshlrev_b32_e32 v4, 2, v4           // M_base = M_base_idx * 4
    
    // Store to LDS rows M_base + 0,1,2,3
    v_lshlrev_b32_e32 v5, 7, v4           // M_base * 128
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v32
    ds_write_b32 v5, v33 offset:128
    ds_write_b32 v5, v34 offset:256
    ds_write_b32 v5, v35 offset:384
    
    // Store to LDS rows M_base + 8,9,10,11
    v_add_u32_e32 v6, 8, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v36
    ds_write_b32 v5, v37 offset:128
    ds_write_b32 v5, v38 offset:256
    ds_write_b32 v5, v39 offset:384
    
    // Store to LDS rows M_base + 16,17,18,19
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v40
    ds_write_b32 v5, v41 offset:128
    ds_write_b32 v5, v42 offset:256
    ds_write_b32 v5, v43 offset:384
    
    // Store to LDS rows M_base + 24,25,26,27
    v_add_u32_e32 v6, 24, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v44
    ds_write_b32 v5, v45 offset:128
    ds_write_b32 v5, v46 offset:256
    ds_write_b32 v5, v47 offset:384
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // Softmax: P = softmax(S)
    // Each thread tid handles one row: row = tid % 32 for tid 0-31
    // and row = tid % 32 for tid 32-63 (duplicate work, but simpler)
    // Actually, let's have 64 threads process 32 rows (2 threads per row)
    // Thread tid processes row tid % 32, loading columns:
    //   tid 0-31: cols 0-15
    //   tid 32-63: cols 16-31
    // Then we need cross-lane reduction
    // ========================================================================
    
    // Simpler approach: each thread processes one row completely
    // tid 0-31 process rows 0-31, tid 32-63 duplicate (or process nothing)
    
    v_and_b32_e32 v1, 31, v0              // row = tid % 32
    
    // Load entire row from LDS (32 floats = 128 bytes)
    v_lshlrev_b32_e32 v2, 7, v1           // row * 128
    v_add_u32_e32 v2, LDS_S, v2
    
    // Load 32 values in 8 dwordx4 loads
    ds_read_b128 v[48:51], v2
    ds_read_b128 v[52:55], v2 offset:16
    ds_read_b128 v[56:59], v2 offset:32
    ds_read_b128 v[60:63], v2 offset:48
    ds_read_b128 v[64:67], v2 offset:64
    ds_read_b128 v[68:71], v2 offset:80
    ds_read_b128 v[72:75], v2 offset:96
    ds_read_b128 v[76:79], v2 offset:112
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // Find max of row (32 values in v[48:79])
    // ========================================================================
    v_max_f32_e32 v80, v48, v49
    v_max_f32_e32 v80, v80, v50
    v_max_f32_e32 v80, v80, v51
    v_max_f32_e32 v80, v80, v52
    v_max_f32_e32 v80, v80, v53
    v_max_f32_e32 v80, v80, v54
    v_max_f32_e32 v80, v80, v55
    v_max_f32_e32 v80, v80, v56
    v_max_f32_e32 v80, v80, v57
    v_max_f32_e32 v80, v80, v58
    v_max_f32_e32 v80, v80, v59
    v_max_f32_e32 v80, v80, v60
    v_max_f32_e32 v80, v80, v61
    v_max_f32_e32 v80, v80, v62
    v_max_f32_e32 v80, v80, v63
    v_max_f32_e32 v80, v80, v64
    v_max_f32_e32 v80, v80, v65
    v_max_f32_e32 v80, v80, v66
    v_max_f32_e32 v80, v80, v67
    v_max_f32_e32 v80, v80, v68
    v_max_f32_e32 v80, v80, v69
    v_max_f32_e32 v80, v80, v70
    v_max_f32_e32 v80, v80, v71
    v_max_f32_e32 v80, v80, v72
    v_max_f32_e32 v80, v80, v73
    v_max_f32_e32 v80, v80, v74
    v_max_f32_e32 v80, v80, v75
    v_max_f32_e32 v80, v80, v76
    v_max_f32_e32 v80, v80, v77
    v_max_f32_e32 v80, v80, v78
    v_max_f32_e32 v80, v80, v79
    // v80 = max of row
    
    // ========================================================================
    // Subtract max and compute exp
    // v_exp_f32 computes 2^x, we need e^x = 2^(x * log2(e))
    // log2(e) = 1.4426950408889634 = 0x3fb8aa3b
    // ========================================================================
    s_mov_b32 s14, 0x3fb8aa3b            // log2(e)
    v_mov_b32_e32 v82, s14
    
    v_sub_f32_e32 v48, v48, v80
    v_sub_f32_e32 v49, v49, v80
    v_sub_f32_e32 v50, v50, v80
    v_sub_f32_e32 v51, v51, v80
    v_sub_f32_e32 v52, v52, v80
    v_sub_f32_e32 v53, v53, v80
    v_sub_f32_e32 v54, v54, v80
    v_sub_f32_e32 v55, v55, v80
    v_sub_f32_e32 v56, v56, v80
    v_sub_f32_e32 v57, v57, v80
    v_sub_f32_e32 v58, v58, v80
    v_sub_f32_e32 v59, v59, v80
    v_sub_f32_e32 v60, v60, v80
    v_sub_f32_e32 v61, v61, v80
    v_sub_f32_e32 v62, v62, v80
    v_sub_f32_e32 v63, v63, v80
    v_sub_f32_e32 v64, v64, v80
    v_sub_f32_e32 v65, v65, v80
    v_sub_f32_e32 v66, v66, v80
    v_sub_f32_e32 v67, v67, v80
    v_sub_f32_e32 v68, v68, v80
    v_sub_f32_e32 v69, v69, v80
    v_sub_f32_e32 v70, v70, v80
    v_sub_f32_e32 v71, v71, v80
    v_sub_f32_e32 v72, v72, v80
    v_sub_f32_e32 v73, v73, v80
    v_sub_f32_e32 v74, v74, v80
    v_sub_f32_e32 v75, v75, v80
    v_sub_f32_e32 v76, v76, v80
    v_sub_f32_e32 v77, v77, v80
    v_sub_f32_e32 v78, v78, v80
    v_sub_f32_e32 v79, v79, v80
    
    // Scale by log2(e) for 2^x -> e^x conversion
    v_mul_f32_e32 v48, v48, v82
    v_mul_f32_e32 v49, v49, v82
    v_mul_f32_e32 v50, v50, v82
    v_mul_f32_e32 v51, v51, v82
    v_mul_f32_e32 v52, v52, v82
    v_mul_f32_e32 v53, v53, v82
    v_mul_f32_e32 v54, v54, v82
    v_mul_f32_e32 v55, v55, v82
    v_mul_f32_e32 v56, v56, v82
    v_mul_f32_e32 v57, v57, v82
    v_mul_f32_e32 v58, v58, v82
    v_mul_f32_e32 v59, v59, v82
    v_mul_f32_e32 v60, v60, v82
    v_mul_f32_e32 v61, v61, v82
    v_mul_f32_e32 v62, v62, v82
    v_mul_f32_e32 v63, v63, v82
    v_mul_f32_e32 v64, v64, v82
    v_mul_f32_e32 v65, v65, v82
    v_mul_f32_e32 v66, v66, v82
    v_mul_f32_e32 v67, v67, v82
    v_mul_f32_e32 v68, v68, v82
    v_mul_f32_e32 v69, v69, v82
    v_mul_f32_e32 v70, v70, v82
    v_mul_f32_e32 v71, v71, v82
    v_mul_f32_e32 v72, v72, v82
    v_mul_f32_e32 v73, v73, v82
    v_mul_f32_e32 v74, v74, v82
    v_mul_f32_e32 v75, v75, v82
    v_mul_f32_e32 v76, v76, v82
    v_mul_f32_e32 v77, v77, v82
    v_mul_f32_e32 v78, v78, v82
    v_mul_f32_e32 v79, v79, v82
    
    // exp(x) via 2^(x*log2(e))
    v_exp_f32_e32 v48, v48
    v_exp_f32_e32 v49, v49
    v_exp_f32_e32 v50, v50
    v_exp_f32_e32 v51, v51
    v_exp_f32_e32 v52, v52
    v_exp_f32_e32 v53, v53
    v_exp_f32_e32 v54, v54
    v_exp_f32_e32 v55, v55
    v_exp_f32_e32 v56, v56
    v_exp_f32_e32 v57, v57
    v_exp_f32_e32 v58, v58
    v_exp_f32_e32 v59, v59
    v_exp_f32_e32 v60, v60
    v_exp_f32_e32 v61, v61
    v_exp_f32_e32 v62, v62
    v_exp_f32_e32 v63, v63
    v_exp_f32_e32 v64, v64
    v_exp_f32_e32 v65, v65
    v_exp_f32_e32 v66, v66
    v_exp_f32_e32 v67, v67
    v_exp_f32_e32 v68, v68
    v_exp_f32_e32 v69, v69
    v_exp_f32_e32 v70, v70
    v_exp_f32_e32 v71, v71
    v_exp_f32_e32 v72, v72
    v_exp_f32_e32 v73, v73
    v_exp_f32_e32 v74, v74
    v_exp_f32_e32 v75, v75
    v_exp_f32_e32 v76, v76
    v_exp_f32_e32 v77, v77
    v_exp_f32_e32 v78, v78
    v_exp_f32_e32 v79, v79
    
    // ========================================================================
    // Sum exp values
    // ========================================================================
    v_add_f32_e32 v81, v48, v49
    v_add_f32_e32 v81, v81, v50
    v_add_f32_e32 v81, v81, v51
    v_add_f32_e32 v81, v81, v52
    v_add_f32_e32 v81, v81, v53
    v_add_f32_e32 v81, v81, v54
    v_add_f32_e32 v81, v81, v55
    v_add_f32_e32 v81, v81, v56
    v_add_f32_e32 v81, v81, v57
    v_add_f32_e32 v81, v81, v58
    v_add_f32_e32 v81, v81, v59
    v_add_f32_e32 v81, v81, v60
    v_add_f32_e32 v81, v81, v61
    v_add_f32_e32 v81, v81, v62
    v_add_f32_e32 v81, v81, v63
    v_add_f32_e32 v81, v81, v64
    v_add_f32_e32 v81, v81, v65
    v_add_f32_e32 v81, v81, v66
    v_add_f32_e32 v81, v81, v67
    v_add_f32_e32 v81, v81, v68
    v_add_f32_e32 v81, v81, v69
    v_add_f32_e32 v81, v81, v70
    v_add_f32_e32 v81, v81, v71
    v_add_f32_e32 v81, v81, v72
    v_add_f32_e32 v81, v81, v73
    v_add_f32_e32 v81, v81, v74
    v_add_f32_e32 v81, v81, v75
    v_add_f32_e32 v81, v81, v76
    v_add_f32_e32 v81, v81, v77
    v_add_f32_e32 v81, v81, v78
    v_add_f32_e32 v81, v81, v79
    // v81 = sum of exp
    
    // ========================================================================
    // Compute 1/sum (reciprocal)
    // CRITICAL: v_rcp_f32 has pipeline hazard - need nops
    // ========================================================================
    v_rcp_f32_e32 v81, v81
    s_nop 7
    s_nop 7
    
    // ========================================================================
    // Normalize: P = exp / sum
    // ========================================================================
    v_mul_f32_e32 v48, v48, v81
    v_mul_f32_e32 v49, v49, v81
    v_mul_f32_e32 v50, v50, v81
    v_mul_f32_e32 v51, v51, v81
    v_mul_f32_e32 v52, v52, v81
    v_mul_f32_e32 v53, v53, v81
    v_mul_f32_e32 v54, v54, v81
    v_mul_f32_e32 v55, v55, v81
    v_mul_f32_e32 v56, v56, v81
    v_mul_f32_e32 v57, v57, v81
    v_mul_f32_e32 v58, v58, v81
    v_mul_f32_e32 v59, v59, v81
    v_mul_f32_e32 v60, v60, v81
    v_mul_f32_e32 v61, v61, v81
    v_mul_f32_e32 v62, v62, v81
    v_mul_f32_e32 v63, v63, v81
    v_mul_f32_e32 v64, v64, v81
    v_mul_f32_e32 v65, v65, v81
    v_mul_f32_e32 v66, v66, v81
    v_mul_f32_e32 v67, v67, v81
    v_mul_f32_e32 v68, v68, v81
    v_mul_f32_e32 v69, v69, v81
    v_mul_f32_e32 v70, v70, v81
    v_mul_f32_e32 v71, v71, v81
    v_mul_f32_e32 v72, v72, v81
    v_mul_f32_e32 v73, v73, v81
    v_mul_f32_e32 v74, v74, v81
    v_mul_f32_e32 v75, v75, v81
    v_mul_f32_e32 v76, v76, v81
    v_mul_f32_e32 v77, v77, v81
    v_mul_f32_e32 v78, v78, v81
    v_mul_f32_e32 v79, v79, v81
    
    // ========================================================================
    // Store P to global memory
    // Only threads 0-31 store (they computed unique rows)
    // ========================================================================
    v_cmp_lt_u32_e64 s[2:3], v0, 32
    s_and_saveexec_b64 s[4:5], s[2:3]
    
    // row * 128 bytes
    v_lshlrev_b32_e32 v2, 7, v1
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v2, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_store_dwordx4 v[10:11], v[48:51]
    flat_store_dwordx4 v[10:11], v[52:55] offset:16
    flat_store_dwordx4 v[10:11], v[56:59] offset:32
    flat_store_dwordx4 v[10:11], v[60:63] offset:48
    flat_store_dwordx4 v[10:11], v[64:67] offset:64
    flat_store_dwordx4 v[10:11], v[68:71] offset:80
    flat_store_dwordx4 v[10:11], v[72:75] offset:96
    flat_store_dwordx4 v[10:11], v[76:79] offset:112
    
    s_mov_b64 exec, s[4:5]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter20integrate_step2_softmaxE, .-_ZN5aiter20integrate_step2_softmaxE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter20integrate_step2_softmaxE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 100
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 84
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter20integrate_step2_softmaxE
    .symbol: _ZN5aiter20integrate_step2_softmaxE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 100
    .agpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - .name: ptr_out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_Q
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_K
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

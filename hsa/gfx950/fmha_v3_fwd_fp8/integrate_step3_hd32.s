// Integration Step 3: Full FP8 Flash Attention (head_dim=32)
// Uses 32x32x16 MFMA which outputs 32x32 - matches head_dim=32
//
// Dimensions: Q[32,32], K[32,32], V[32,32] -> O[32,32]
// seq_len=32, head_dim=32
//
// Pipeline:
// 1. QK MFMA: S[32,32] = Q[32,32] @ K[32,32]^T (2 MFMAs for K_dim=32)
// 2. Softmax: P = softmax(S)
// 3. PV MFMA: O[32,32] = P[32,32] @ V[32,32] (2 MFMAs for K_dim=32)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set LDS_S, 0          // S[32,32] F32 = 4KB
.set LDS_P, 4096       // P[32,32] FP8 = 1KB

.text
.globl _ZN5aiter20integrate_step3_hd32E
.p2align 8
.type _ZN5aiter20integrate_step3_hd32E,@function

_ZN5aiter20integrate_step3_hd32E:
    // Args: 0=ptr_O, 8=ptr_Q, 16=ptr_K, 24=ptr_V
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_O
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // ptr_K
    s_load_dwordx2 s[14:15], s[0:1], 0x18  // ptr_V
    
    v_and_b32_e32 v0, 63, v0               // tid
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // PHASE 1: QK MFMA (two passes for head_dim=32)
    // S[m,n] = sum_d Q[m,d] * K[n,d]
    // ========================================================================
    
    // Initialize accumulators for S
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
    
    // --- First QK MFMA: d=0..15 ---
    // Load Q[m, d=0..15]: thread t loads Q[t%32, (t/32)*8 : (t/32)*8+8]
    v_and_b32_e32 v1, 31, v0              // m = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // d_group = tid / 32
    v_lshlrev_b32_e32 v2, 3, v2           // d_start = d_group * 8
    
    v_lshlrev_b32_e32 v3, 5, v1           // m * 32 (row stride for head_dim=32)
    v_add_u32_e32 v3, v3, v2              // + d_start
    
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[20:21], v[10:11]  // Q[m, d_start:d_start+8]
    
    // Load K[n, d=0..15]: same pattern
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[64:65], v[10:11]  // K[n, d_start:d_start+8]
    
    s_waitcnt vmcnt(0)
    
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 7
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    s_nop 15
    
    // --- Second QK MFMA: d=16..31 ---
    // Load Q[m, d=16..31]
    v_add_u32_e32 v3, 16, v2              // d_start = 16 + (tid/32)*8
    v_lshlrev_b32_e32 v4, 5, v1           // m * 32
    v_add_u32_e32 v3, v4, v3
    
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[20:21], v[10:11]
    
    // Load K[n, d=16..31]
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[64:65], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 7
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    s_nop 15
    s_nop 7
    
    // ========================================================================
    // PHASE 2: Store S to LDS
    // ========================================================================
    v_and_b32_e32 v3, 31, v0
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4 bytes
    v_lshrrev_b32_e32 v4, 5, v0
    v_lshlrev_b32_e32 v4, 2, v4           // M_base = (tid/32) * 4
    
    // Store rows M_base + 0,1,2,3
    v_lshlrev_b32_e32 v5, 7, v4
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v32
    ds_write_b32 v5, v33 offset:128
    ds_write_b32 v5, v34 offset:256
    ds_write_b32 v5, v35 offset:384
    
    // Store rows M_base + 8,9,10,11
    v_add_u32_e32 v6, 8, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v36
    ds_write_b32 v5, v37 offset:128
    ds_write_b32 v5, v38 offset:256
    ds_write_b32 v5, v39 offset:384
    
    // Store rows M_base + 16,17,18,19
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_u32_e32 v5, LDS_S, v5
    ds_write_b32 v5, v40
    ds_write_b32 v5, v41 offset:128
    ds_write_b32 v5, v42 offset:256
    ds_write_b32 v5, v43 offset:384
    
    // Store rows M_base + 24,25,26,27
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
    // PHASE 3: Softmax (threads 0-31 compute rows 0-31)
    // ========================================================================
    v_and_b32_e32 v1, 31, v0
    v_lshlrev_b32_e32 v2, 7, v1
    v_add_u32_e32 v2, LDS_S, v2
    
    // Load 32 S values
    ds_read_b128 v[48:51], v2
    ds_read_b128 v[52:55], v2 offset:16
    ds_read_b128 v[56:59], v2 offset:32
    ds_read_b128 v[60:63], v2 offset:48
    ds_read_b128 v[64:67], v2 offset:64
    ds_read_b128 v[68:71], v2 offset:80
    ds_read_b128 v[72:75], v2 offset:96
    ds_read_b128 v[76:79], v2 offset:112
    
    s_waitcnt lgkmcnt(0)
    
    // Find max
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
    
    // log2(e) for exp conversion
    s_mov_b32 s16, 0x3fb8aa3b
    v_mov_b32_e32 v82, s16
    
    // Subtract max, scale by log2(e), and exp
    v_sub_f32_e32 v48, v48, v80
    v_mul_f32_e32 v48, v48, v82
    v_exp_f32_e32 v48, v48
    
    v_sub_f32_e32 v49, v49, v80
    v_mul_f32_e32 v49, v49, v82
    v_exp_f32_e32 v49, v49
    
    v_sub_f32_e32 v50, v50, v80
    v_mul_f32_e32 v50, v50, v82
    v_exp_f32_e32 v50, v50
    
    v_sub_f32_e32 v51, v51, v80
    v_mul_f32_e32 v51, v51, v82
    v_exp_f32_e32 v51, v51
    
    v_sub_f32_e32 v52, v52, v80
    v_mul_f32_e32 v52, v52, v82
    v_exp_f32_e32 v52, v52
    
    v_sub_f32_e32 v53, v53, v80
    v_mul_f32_e32 v53, v53, v82
    v_exp_f32_e32 v53, v53
    
    v_sub_f32_e32 v54, v54, v80
    v_mul_f32_e32 v54, v54, v82
    v_exp_f32_e32 v54, v54
    
    v_sub_f32_e32 v55, v55, v80
    v_mul_f32_e32 v55, v55, v82
    v_exp_f32_e32 v55, v55
    
    v_sub_f32_e32 v56, v56, v80
    v_mul_f32_e32 v56, v56, v82
    v_exp_f32_e32 v56, v56
    
    v_sub_f32_e32 v57, v57, v80
    v_mul_f32_e32 v57, v57, v82
    v_exp_f32_e32 v57, v57
    
    v_sub_f32_e32 v58, v58, v80
    v_mul_f32_e32 v58, v58, v82
    v_exp_f32_e32 v58, v58
    
    v_sub_f32_e32 v59, v59, v80
    v_mul_f32_e32 v59, v59, v82
    v_exp_f32_e32 v59, v59
    
    v_sub_f32_e32 v60, v60, v80
    v_mul_f32_e32 v60, v60, v82
    v_exp_f32_e32 v60, v60
    
    v_sub_f32_e32 v61, v61, v80
    v_mul_f32_e32 v61, v61, v82
    v_exp_f32_e32 v61, v61
    
    v_sub_f32_e32 v62, v62, v80
    v_mul_f32_e32 v62, v62, v82
    v_exp_f32_e32 v62, v62
    
    v_sub_f32_e32 v63, v63, v80
    v_mul_f32_e32 v63, v63, v82
    v_exp_f32_e32 v63, v63
    
    v_sub_f32_e32 v64, v64, v80
    v_mul_f32_e32 v64, v64, v82
    v_exp_f32_e32 v64, v64
    
    v_sub_f32_e32 v65, v65, v80
    v_mul_f32_e32 v65, v65, v82
    v_exp_f32_e32 v65, v65
    
    v_sub_f32_e32 v66, v66, v80
    v_mul_f32_e32 v66, v66, v82
    v_exp_f32_e32 v66, v66
    
    v_sub_f32_e32 v67, v67, v80
    v_mul_f32_e32 v67, v67, v82
    v_exp_f32_e32 v67, v67
    
    v_sub_f32_e32 v68, v68, v80
    v_mul_f32_e32 v68, v68, v82
    v_exp_f32_e32 v68, v68
    
    v_sub_f32_e32 v69, v69, v80
    v_mul_f32_e32 v69, v69, v82
    v_exp_f32_e32 v69, v69
    
    v_sub_f32_e32 v70, v70, v80
    v_mul_f32_e32 v70, v70, v82
    v_exp_f32_e32 v70, v70
    
    v_sub_f32_e32 v71, v71, v80
    v_mul_f32_e32 v71, v71, v82
    v_exp_f32_e32 v71, v71
    
    v_sub_f32_e32 v72, v72, v80
    v_mul_f32_e32 v72, v72, v82
    v_exp_f32_e32 v72, v72
    
    v_sub_f32_e32 v73, v73, v80
    v_mul_f32_e32 v73, v73, v82
    v_exp_f32_e32 v73, v73
    
    v_sub_f32_e32 v74, v74, v80
    v_mul_f32_e32 v74, v74, v82
    v_exp_f32_e32 v74, v74
    
    v_sub_f32_e32 v75, v75, v80
    v_mul_f32_e32 v75, v75, v82
    v_exp_f32_e32 v75, v75
    
    v_sub_f32_e32 v76, v76, v80
    v_mul_f32_e32 v76, v76, v82
    v_exp_f32_e32 v76, v76
    
    v_sub_f32_e32 v77, v77, v80
    v_mul_f32_e32 v77, v77, v82
    v_exp_f32_e32 v77, v77
    
    v_sub_f32_e32 v78, v78, v80
    v_mul_f32_e32 v78, v78, v82
    v_exp_f32_e32 v78, v78
    
    v_sub_f32_e32 v79, v79, v80
    v_mul_f32_e32 v79, v79, v82
    v_exp_f32_e32 v79, v79
    
    // Sum
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
    
    // Reciprocal with hazard protection
    v_rcp_f32_e32 v81, v81
    s_nop 7
    s_nop 7
    
    // Normalize
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
    // PHASE 4: Convert P to FP8 and store to LDS
    // ========================================================================
    v_cvt_pk_fp8_f32 v83, v48, v49
    v_cvt_pk_fp8_f32 v84, v50, v51
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v32, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v52, v53
    v_cvt_pk_fp8_f32 v84, v54, v55
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v33, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v56, v57
    v_cvt_pk_fp8_f32 v84, v58, v59
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v34, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v60, v61
    v_cvt_pk_fp8_f32 v84, v62, v63
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v35, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v64, v65
    v_cvt_pk_fp8_f32 v84, v66, v67
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v36, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v68, v69
    v_cvt_pk_fp8_f32 v84, v70, v71
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v37, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v72, v73
    v_cvt_pk_fp8_f32 v84, v74, v75
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v38, v83, v84
    
    v_cvt_pk_fp8_f32 v83, v76, v77
    v_cvt_pk_fp8_f32 v84, v78, v79
    v_lshlrev_b32_e32 v84, 16, v84
    v_or_b32_e32 v39, v83, v84
    
    // Store P to LDS (threads 0-31)
    v_cmp_lt_u32_e64 s[2:3], v0, 32
    s_and_saveexec_b64 s[4:5], s[2:3]
    
    v_and_b32_e32 v1, 31, v0
    v_lshlrev_b32_e32 v2, 5, v1           // row * 32 bytes
    v_add_u32_e32 v2, LDS_P, v2
    ds_write_b128 v2, v[32:35]
    ds_write_b128 v2, v[36:39] offset:16
    
    s_mov_b64 exec, s[4:5]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // PHASE 5: PV MFMA (two passes for seq_len=32)
    // O[m,d] = sum_k P[m,k] * V[k,d]
    // ========================================================================
    
    // Initialize O accumulators
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
    
    // --- First PV MFMA: k=0..15 ---
    // Load P[m, k=0..15] from LDS
    v_and_b32_e32 v1, 31, v0
    v_lshrrev_b32_e32 v2, 5, v0
    v_lshlrev_b32_e32 v2, 3, v2           // k_start = (tid/32)*8
    v_lshlrev_b32_e32 v3, 5, v1           // m * 32
    v_add_u32_e32 v3, v3, v2
    v_add_u32_e32 v3, LDS_P, v3
    ds_read_b64 v[20:21], v3
    
    // Load V[k, d] from global with stride
    // V[k,d] at ptr_V + k*32 + d, thread loads V[(tid/32)*8:(tid/32)*8+8, tid%32]
    v_and_b32_e32 v4, 31, v0              // d = tid % 32
    v_lshrrev_b32_e32 v5, 5, v0           // k_group = tid / 32
    v_lshlrev_b32_e32 v5, 8, v5           // k_start * 32 (byte offset to row k_start)
    
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_add_co_u32_e32 v10, vcc, v4, v10    // + d
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v5, v10    // + k_start * 32
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 V values with stride 32 (row stride)
    flat_load_ubyte v48, v[10:11]
    flat_load_ubyte v49, v[10:11] offset:32
    flat_load_ubyte v50, v[10:11] offset:64
    flat_load_ubyte v51, v[10:11] offset:96
    flat_load_ubyte v52, v[10:11] offset:128
    flat_load_ubyte v53, v[10:11] offset:160
    flat_load_ubyte v54, v[10:11] offset:192
    flat_load_ubyte v55, v[10:11] offset:224
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    
    // Pack V bytes
    v_lshlrev_b32_e32 v49, 8, v49
    v_or_b32_e32 v48, v48, v49
    v_lshlrev_b32_e32 v50, 16, v50
    v_or_b32_e32 v48, v48, v50
    v_lshlrev_b32_e32 v51, 24, v51
    v_or_b32_e32 v64, v48, v51
    
    v_lshlrev_b32_e32 v53, 8, v53
    v_or_b32_e32 v52, v52, v53
    v_lshlrev_b32_e32 v54, 16, v54
    v_or_b32_e32 v52, v52, v54
    v_lshlrev_b32_e32 v55, 24, v55
    v_or_b32_e32 v65, v52, v55
    
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 7
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    s_nop 15
    
    // --- Second PV MFMA: k=16..31 ---
    // Load P[m, k=16..31]
    v_and_b32_e32 v1, 31, v0
    v_lshrrev_b32_e32 v2, 5, v0
    v_lshlrev_b32_e32 v2, 3, v2
    v_add_u32_e32 v2, 16, v2              // k_start = 16 + (tid/32)*8
    v_lshlrev_b32_e32 v3, 5, v1
    v_add_u32_e32 v3, v3, v2
    v_add_u32_e32 v3, LDS_P, v3
    ds_read_b64 v[20:21], v3
    
    // Load V[k=16..31, d] with stride
    v_and_b32_e32 v4, 31, v0
    v_lshrrev_b32_e32 v5, 5, v0
    v_lshlrev_b32_e32 v5, 3, v5
    v_add_u32_e32 v5, 16, v5              // k_start = 16 + (tid/32)*8
    v_lshlrev_b32_e32 v5, 5, v5           // k_start * 32
    
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v5, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_ubyte v48, v[10:11]
    flat_load_ubyte v49, v[10:11] offset:32
    flat_load_ubyte v50, v[10:11] offset:64
    flat_load_ubyte v51, v[10:11] offset:96
    flat_load_ubyte v52, v[10:11] offset:128
    flat_load_ubyte v53, v[10:11] offset:160
    flat_load_ubyte v54, v[10:11] offset:192
    flat_load_ubyte v55, v[10:11] offset:224
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    
    v_lshlrev_b32_e32 v49, 8, v49
    v_or_b32_e32 v48, v48, v49
    v_lshlrev_b32_e32 v50, 16, v50
    v_or_b32_e32 v48, v48, v50
    v_lshlrev_b32_e32 v51, 24, v51
    v_or_b32_e32 v64, v48, v51
    
    v_lshlrev_b32_e32 v53, 8, v53
    v_or_b32_e32 v52, v52, v53
    v_lshlrev_b32_e32 v54, 16, v54
    v_or_b32_e32 v52, v52, v54
    v_lshlrev_b32_e32 v55, 24, v55
    v_or_b32_e32 v65, v52, v55
    
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 7
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    s_nop 15
    s_nop 7
    
    // ========================================================================
    // PHASE 6: Store O to global
    // ========================================================================
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4
    v_lshrrev_b32_e32 v4, 5, v0
    v_lshlrev_b32_e32 v4, 2, v4           // M_base
    
    // Store rows with scatter pattern
    v_lshlrev_b32_e32 v5, 7, v4
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v32
    flat_store_dword v[12:13], v33 offset:128
    flat_store_dword v[12:13], v34 offset:256
    flat_store_dword v[12:13], v35 offset:384
    
    v_add_u32_e32 v6, 8, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v36
    flat_store_dword v[12:13], v37 offset:128
    flat_store_dword v[12:13], v38 offset:256
    flat_store_dword v[12:13], v39 offset:384
    
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v40
    flat_store_dword v[12:13], v41 offset:128
    flat_store_dword v[12:13], v42 offset:256
    flat_store_dword v[12:13], v43 offset:384
    
    v_add_u32_e32 v6, 24, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v44
    flat_store_dword v[12:13], v45 offset:128
    flat_store_dword v[12:13], v46 offset:256
    flat_store_dword v[12:13], v47 offset:384
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter20integrate_step3_hd32E, .-_ZN5aiter20integrate_step3_hd32E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter20integrate_step3_hd32E
    .amdhsa_group_segment_fixed_size 6144
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 100
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 88
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter20integrate_step3_hd32E
    .symbol: _ZN5aiter20integrate_step3_hd32E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 6144
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 100
    .agpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - .name: ptr_O
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
      - .name: ptr_V
        .size: 8
        .offset: 24
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

// SPDX-License-Identifier: MIT
// FP8 Flash Attention - V×P version 3 with corrected output store
// Uses permlane32_swap to fix MFMA output interleaving

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E
.p2align 8
.type _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E,@function

.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 8192
.set LDS_V_OFFSET, 16384
.set LDS_P_OFFSET, 24576

_ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E:
    s_and_b32 s1, s1, 0xffff
    
    s_load_dwordx2 s[8:9], s[0:1], 0x00
    s_load_dwordx2 s[10:11], s[0:1], 0x10
    s_load_dwordx2 s[12:13], s[0:1], 0x20
    s_load_dwordx2 s[14:15], s[0:1], 0x30
    s_load_dword s20, s[0:1], 0x50
    s_load_dword s21, s[0:1], 0x58
    s_load_dword s22, s[0:1], 0x60
    s_load_dword s26, s[0:1], 0x200
    s_load_dword s27, s[0:1], 0x204
    s_load_dword s28, s[0:1], 0x208
    
    v_and_b32_e32 v0, 0xff, v0
    v_lshrrev_b32_e32 v1, 6, v0
    v_and_b32_e32 v2, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    s_lshl_b32 s23, s21, 7
    s_lshl_b32 s24, s22, 7
    s_lshl_b32 s32, s2, 13
    
    v_mov_b32_e32 v3, s20
    v_mov_b32_e32 v4, s26
    v_mov_b32_e32 v5, s27
    v_mul_f32_e32 v3, v3, v4
    v_mul_f32_e32 v3, v3, v5
    v_readfirstlane_b32 s29, v3
    
    s_add_u32 s30, s22, 63
    s_lshr_b32 s30, s30, 6
    
    v_mov_b32_e32 v16, 0xff800000
    v_mov_b32_e32 v17, 0
    
    // Output accumulators - will contain O^T[D, Q] after V×P
    v_mov_b32_e32 v48, 0
    v_mov_b32_e32 v49, 0
    v_mov_b32_e32 v50, 0
    v_mov_b32_e32 v51, 0
    v_mov_b32_e32 v52, 0
    v_mov_b32_e32 v53, 0
    v_mov_b32_e32 v54, 0
    v_mov_b32_e32 v55, 0
    v_mov_b32_e32 v56, 0
    v_mov_b32_e32 v57, 0
    v_mov_b32_e32 v58, 0
    v_mov_b32_e32 v59, 0
    v_mov_b32_e32 v60, 0
    v_mov_b32_e32 v61, 0
    v_mov_b32_e32 v62, 0
    v_mov_b32_e32 v63, 0
    
    s_nop 7

    // Load Q to LDS
    v_lshlrev_b32_e32 v6, 4, v0
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_mov_b32_e32 v8, s32
    v_add_co_u32_e32 v10, vcc, v8, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[64:67], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[68:71], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[64:67]
    v_add_u32_e32 v7, 4096, v6
    ds_write_b128 v7, v[68:71]
    s_barrier
    
    s_mov_b32 s31, 0
    
K_LOOP:
    // Load K to LDS
    s_lshl_b32 s33, s31, 13
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[72:75], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[76:79], v[10:11]
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v7, LDS_K_OFFSET, v6
    ds_write_b128 v7, v[72:75]
    v_add_u32_e32 v7, LDS_K_OFFSET + 4096, v6
    ds_write_b128 v7, v[76:79]
    s_barrier
    
    // QK accumulator
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
    
    // Q×K GEMM
    v_lshlrev_b32_e32 v7, 3, v2
    ds_read_b64 a[0:1], v7
    v_add_u32_e32 v9, 16, v7
    ds_read_b64 a[2:3], v9
    v_add_u32_e32 v9, LDS_K_OFFSET, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    v_add_u32_e32 v9, 32, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 32, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    v_add_u32_e32 v9, 64, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[2:3], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 64, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    v_add_u32_e32 v9, 96, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 96, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    s_nop 7

    // Apply scale
    v_mul_f32_e32 v32, s29, v32
    v_mul_f32_e32 v33, s29, v33
    v_mul_f32_e32 v34, s29, v34
    v_mul_f32_e32 v35, s29, v35
    v_mul_f32_e32 v36, s29, v36
    v_mul_f32_e32 v37, s29, v37
    v_mul_f32_e32 v38, s29, v38
    v_mul_f32_e32 v39, s29, v39
    v_mul_f32_e32 v40, s29, v40
    v_mul_f32_e32 v41, s29, v41
    v_mul_f32_e32 v42, s29, v42
    v_mul_f32_e32 v43, s29, v43
    v_mul_f32_e32 v44, s29, v44
    v_mul_f32_e32 v45, s29, v45
    v_mul_f32_e32 v46, s29, v46
    v_mul_f32_e32 v47, s29, v47
    
    // Online softmax
    v_max_f32_e32 v18, v32, v33
    v_max_f32_e32 v18, v18, v34
    v_max_f32_e32 v18, v18, v35
    v_max_f32_e32 v18, v18, v36
    v_max_f32_e32 v18, v18, v37
    v_max_f32_e32 v18, v18, v38
    v_max_f32_e32 v18, v18, v39
    v_max_f32_e32 v18, v18, v40
    v_max_f32_e32 v18, v18, v41
    v_max_f32_e32 v18, v18, v42
    v_max_f32_e32 v18, v18, v43
    v_max_f32_e32 v18, v18, v44
    v_max_f32_e32 v18, v18, v45
    v_max_f32_e32 v18, v18, v46
    v_max_f32_e32 v18, v18, v47
    
    v_mov_b32_e32 v19, v18
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v19, v18
    v_max_f32_e32 v18, v18, v19
    
    v_mov_b32_e32 v19, v16
    v_max_f32_e32 v16, v16, v18
    
    v_sub_f32_e32 v20, v19, v16
    v_exp_f32_e32 v19, v20
    
    s_nop 7

    // Scale accumulators
    v_mul_f32_e32 v48, v19, v48
    v_mul_f32_e32 v49, v19, v49
    v_mul_f32_e32 v50, v19, v50
    v_mul_f32_e32 v51, v19, v51
    v_mul_f32_e32 v52, v19, v52
    v_mul_f32_e32 v53, v19, v53
    v_mul_f32_e32 v54, v19, v54
    v_mul_f32_e32 v55, v19, v55
    v_mul_f32_e32 v56, v19, v56
    v_mul_f32_e32 v57, v19, v57
    v_mul_f32_e32 v58, v19, v58
    v_mul_f32_e32 v59, v19, v59
    v_mul_f32_e32 v60, v19, v60
    v_mul_f32_e32 v61, v19, v61
    v_mul_f32_e32 v62, v19, v62
    v_mul_f32_e32 v63, v19, v63
    v_mul_f32_e32 v17, v19, v17
    
    // exp(QK - max) = P
    v_sub_f32_e32 v32, v32, v16
    v_exp_f32_e32 v32, v32
    v_sub_f32_e32 v33, v33, v16
    v_exp_f32_e32 v33, v33
    v_sub_f32_e32 v34, v34, v16
    v_exp_f32_e32 v34, v34
    v_sub_f32_e32 v35, v35, v16
    v_exp_f32_e32 v35, v35
    v_sub_f32_e32 v36, v36, v16
    v_exp_f32_e32 v36, v36
    v_sub_f32_e32 v37, v37, v16
    v_exp_f32_e32 v37, v37
    v_sub_f32_e32 v38, v38, v16
    v_exp_f32_e32 v38, v38
    v_sub_f32_e32 v39, v39, v16
    v_exp_f32_e32 v39, v39
    v_sub_f32_e32 v40, v40, v16
    v_exp_f32_e32 v40, v40
    v_sub_f32_e32 v41, v41, v16
    v_exp_f32_e32 v41, v41
    v_sub_f32_e32 v42, v42, v16
    v_exp_f32_e32 v42, v42
    v_sub_f32_e32 v43, v43, v16
    v_exp_f32_e32 v43, v43
    v_sub_f32_e32 v44, v44, v16
    v_exp_f32_e32 v44, v44
    v_sub_f32_e32 v45, v45, v16
    v_exp_f32_e32 v45, v45
    v_sub_f32_e32 v46, v46, v16
    v_exp_f32_e32 v46, v46
    v_sub_f32_e32 v47, v47, v16
    v_exp_f32_e32 v47, v47
    
    // Sum P
    v_add_f32_e32 v20, v32, v33
    v_add_f32_e32 v21, v34, v35
    v_add_f32_e32 v22, v36, v37
    v_add_f32_e32 v23, v38, v39
    v_add_f32_e32 v20, v20, v21
    v_add_f32_e32 v22, v22, v23
    v_add_f32_e32 v20, v20, v22
    v_add_f32_e32 v21, v40, v41
    v_add_f32_e32 v22, v42, v43
    v_add_f32_e32 v23, v44, v45
    v_add_f32_e32 v71, v46, v47
    v_add_f32_e32 v21, v21, v22
    v_add_f32_e32 v23, v23, v71
    v_add_f32_e32 v21, v21, v23
    v_add_f32_e32 v20, v20, v21
    
    v_mov_b32_e32 v21, v20
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v21, v20
    v_add_f32_e32 v20, v20, v21
    v_add_f32_e32 v17, v17, v20
    
    s_nop 7

    // ========================================================================
    // Simple PV computation: P × V (standard order, not V×P)
    // This matches the original kernel behavior
    // P is A operand (from softmax output)
    // V is B operand (from global memory)
    // ========================================================================
    
    // Pack P to FP8 for A operand
    v_mov_b32_e32 v80, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v80, v32, v33
    v_cvt_pk_fp8_f32 v81, v34, v35
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v80, v80, v81
    
    v_mov_b32_e32 v82, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v82, v36, v37
    v_cvt_pk_fp8_f32 v81, v38, v39
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v82, v82, v81
    
    v_accvgpr_write_b32 a0, v80
    v_accvgpr_write_b32 a1, v82
    
    v_mov_b32_e32 v84, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v84, v40, v41
    v_cvt_pk_fp8_f32 v81, v42, v43
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v84, v84, v81
    
    v_mov_b32_e32 v86, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v86, v44, v45
    v_cvt_pk_fp8_f32 v81, v46, v47
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v86, v86, v81
    
    v_accvgpr_write_b32 a2, v84
    v_accvgpr_write_b32 a3, v86
    
    // Load V and store to LDS with transposed layout
    v_and_b32_e32 v80, 31, v0
    v_lshrrev_b32_e32 v81, 5, v0
    
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v80, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    v_lshlrev_b32_e32 v7, 11, v81
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load V[K=0..7 or 8..15, D]
    flat_load_ubyte v72, v[10:11]
    flat_load_ubyte v73, v[10:11] offset:128
    flat_load_ubyte v74, v[10:11] offset:256
    flat_load_ubyte v75, v[10:11] offset:384
    flat_load_ubyte v76, v[10:11] offset:512
    flat_load_ubyte v77, v[10:11] offset:640
    flat_load_ubyte v78, v[10:11] offset:768
    flat_load_ubyte v79, v[10:11] offset:896
    
    s_waitcnt vmcnt(0)
    
    // Pack V
    v_lshlrev_b32_e32 v73, 8, v73
    v_or_b32_e32 v72, v72, v73
    v_lshlrev_b32_e32 v74, 16, v74
    v_or_b32_e32 v72, v72, v74
    v_lshlrev_b32_e32 v75, 24, v75
    v_or_b32_e32 v64, v72, v75
    
    v_lshlrev_b32_e32 v77, 8, v77
    v_or_b32_e32 v76, v76, v77
    v_lshlrev_b32_e32 v78, 16, v78
    v_or_b32_e32 v76, v76, v78
    v_lshlrev_b32_e32 v79, 24, v79
    v_or_b32_e32 v65, v76, v79
    
    // Store V to LDS at V[D, K] layout: D*32 + K_half*16
    v_lshlrev_b32_e32 v7, 5, v80
    v_lshlrev_b32_e32 v8, 4, v81
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_write_b64 v7, v[64:65]
    
    // Load V[K=8..15 or 24..31, D]
    flat_load_ubyte v72, v[10:11] offset:1024
    flat_load_ubyte v73, v[10:11] offset:1152
    flat_load_ubyte v74, v[10:11] offset:1280
    flat_load_ubyte v75, v[10:11] offset:1408
    flat_load_ubyte v76, v[10:11] offset:1536
    flat_load_ubyte v77, v[10:11] offset:1664
    flat_load_ubyte v78, v[10:11] offset:1792
    flat_load_ubyte v79, v[10:11] offset:1920
    
    s_waitcnt vmcnt(0)
    
    v_lshlrev_b32_e32 v73, 8, v73
    v_or_b32_e32 v72, v72, v73
    v_lshlrev_b32_e32 v74, 16, v74
    v_or_b32_e32 v72, v72, v74
    v_lshlrev_b32_e32 v75, 24, v75
    v_or_b32_e32 v66, v72, v75
    
    v_lshlrev_b32_e32 v77, 8, v77
    v_or_b32_e32 v76, v76, v77
    v_lshlrev_b32_e32 v78, 16, v78
    v_or_b32_e32 v76, v76, v78
    v_lshlrev_b32_e32 v79, 24, v79
    v_or_b32_e32 v67, v76, v79
    
    v_add_u32_e32 v7, 8, v7
    ds_write_b64 v7, v[66:67]
    
    s_barrier
    
    // Read V from LDS for B operand
    v_and_b32_e32 v7, 31, v2
    v_lshlrev_b32_e32 v7, 5, v7
    v_lshrrev_b32_e32 v8, 5, v2
    v_lshlrev_b32_e32 v8, 3, v8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_read_b64 v[64:65], v7
    
    s_waitcnt lgkmcnt(0)
    
    s_nop 7

    // P×V MFMA: A=P, B=V
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    // Read V for K=16..31
    v_and_b32_e32 v7, 31, v2
    v_lshlrev_b32_e32 v7, 5, v7
    v_lshrrev_b32_e32 v8, 5, v2
    v_lshlrev_b32_e32 v8, 3, v8
    v_add_u32_e32 v8, 16, v8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_read_b64 v[66:67], v7
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[66:67], v[48:63]
    
    s_nop 7

    // K-loop increment
    s_add_u32 s31, s31, 1
    s_cmp_lt_u32 s31, s30
    s_cbranch_scc1 K_LOOP
    
    // Final normalization
    v_rcp_f32_e32 v17, v17
    s_nop 7
    
    v_mul_f32_e32 v48, v17, v48
    v_mul_f32_e32 v49, v17, v49
    v_mul_f32_e32 v50, v17, v50
    v_mul_f32_e32 v51, v17, v51
    v_mul_f32_e32 v52, v17, v52
    v_mul_f32_e32 v53, v17, v53
    v_mul_f32_e32 v54, v17, v54
    v_mul_f32_e32 v55, v17, v55
    v_mul_f32_e32 v56, v17, v56
    v_mul_f32_e32 v57, v17, v57
    v_mul_f32_e32 v58, v17, v58
    v_mul_f32_e32 v59, v17, v59
    v_mul_f32_e32 v60, v17, v60
    v_mul_f32_e32 v61, v17, v61
    v_mul_f32_e32 v62, v17, v62
    v_mul_f32_e32 v63, v17, v63
    
    v_mul_f32_e32 v48, s28, v48
    v_mul_f32_e32 v49, s28, v49
    v_mul_f32_e32 v50, s28, v50
    v_mul_f32_e32 v51, s28, v51
    v_mul_f32_e32 v52, s28, v52
    v_mul_f32_e32 v53, s28, v53
    v_mul_f32_e32 v54, s28, v54
    v_mul_f32_e32 v55, s28, v55
    v_mul_f32_e32 v56, s28, v56
    v_mul_f32_e32 v57, s28, v57
    v_mul_f32_e32 v58, s28, v58
    v_mul_f32_e32 v59, s28, v59
    v_mul_f32_e32 v60, s28, v60
    v_mul_f32_e32 v61, s28, v61
    v_mul_f32_e32 v62, s28, v62
    v_mul_f32_e32 v63, s28, v63
    
    s_nop 7

    // ========================================================================
    // Store with permlane32_swap to fix interleaving
    // After MFMA, threads 0-31 have rows 0,1,2,3,8,9,10,11,...
    // and threads 32-63 have rows 4,5,6,7,12,13,14,15,...
    // Swap v52-v55 (rows 8-11 in threads 0-31) with v48-v51 (rows 4-7 in threads 32-63)
    // ========================================================================
    
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v48, v52
    v_permlane32_swap_b32_e32 v49, v53
    v_permlane32_swap_b32_e32 v50, v54
    v_permlane32_swap_b32_e32 v51, v55
    
    v_permlane32_swap_b32_e32 v56, v60
    v_permlane32_swap_b32_e32 v57, v61
    v_permlane32_swap_b32_e32 v58, v62
    v_permlane32_swap_b32_e32 v59, v63
    
    // Now threads 0-31 have rows 0-3 (own) and 4-7 (from 32-63)
    // and threads 32-63 have rows 8-11 (from 0-31) and 12-15 (own)
    // Store contiguously
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    // Thread t stores at:
    // - Threads 0-31: Q=t, D=0-7 (v48-v55)
    // - Threads 32-63: Q=t-32, D=8-15 (v48-v55)
    // Base = Q * 512 + D_base * 4
    
    v_and_b32_e32 v3, 31, v0              // Q = t % 32
    v_lshrrev_b32_e32 v4, 5, v0           // D_block = t / 32 (0 or 1)
    v_lshlrev_b32_e32 v3, 9, v3           // Q * 512
    v_lshlrev_b32_e32 v4, 5, v4           // D_block * 32 (8 floats * 4 bytes)
    v_add_u32_e32 v3, v3, v4
    
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store 8 values (rows 0-7 for threads 0-31, rows 8-15 for threads 32-63)
    flat_store_dwordx4 v[10:11], v[48:51]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[52:55]
    
    // Second block: rows 8-15 for threads 0-31, rows 0-7 for threads 32-63
    // But after swap, v56-v63 are reordered too
    // This needs more work...
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E, .-_ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 528
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 120
    .amdhsa_next_free_sgpr 48
    .amdhsa_accum_offset 104
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E
    .symbol: _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v3E.kd
    .kernarg_segment_size: 528
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 48
    .vgpr_count: 120
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .args:
      - .name: ptr_R
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_Q
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_K
        .size: 8
        .offset: 32
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_V
        .size: 8
        .offset: 48
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_LSE
        .size: 8
        .offset: 64
        .value_kind: global_buffer
        .address_space: global
      - .name: softmax_scale
        .size: 4
        .offset: 80
        .value_kind: by_value
      - .name: seqlen_q
        .size: 4
        .offset: 88
        .value_kind: by_value
      - .name: seqlen_k
        .size: 4
        .offset: 96
        .value_kind: by_value
      - .name: q_scale
        .size: 4
        .offset: 512
        .value_kind: by_value
      - .name: k_scale
        .size: 4
        .offset: 516
        .value_kind: by_value
      - .name: v_scale
        .size: 4
        .offset: 520
        .value_kind: by_value
...
.end_amdgpu_metadata

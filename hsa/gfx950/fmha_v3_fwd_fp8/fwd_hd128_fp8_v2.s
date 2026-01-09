// FP8 Flash Attention Forward Kernel (Clean Version)
// Target: gfx950
// Head dim: 128, Single tile (32x32 QK), Single wave (64 threads)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter18fmha_fwd_hd128_fp8E
.type _ZN5aiter18fmha_fwd_hd128_fp8E, @function
.p2align 8

.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 2048
.set LDS_V_OFFSET, 4096

_ZN5aiter18fmha_fwd_hd128_fp8E:
    // ========================================================================
    // KERNEL ARGUMENTS (loaded from kernarg segment)
    // ========================================================================
    // arg0  (offset 0):   output pointer
    // arg1  (offset 16):  Q pointer
    // arg2  (offset 32):  K pointer
    // arg3  (offset 48):  V pointer
    // arg4  (offset 64):  lse pointer
    // arg5  (offset 80):  scale (float)
    // arg6  (offset 88):  seqlen_q (uint32)
    // arg7  (offset 96):  seqlen_k (uint32)
    // arg8  (offset 512): q_scale (float)
    // arg9  (offset 516): k_scale (float)
    // arg10 (offset 520): v_scale (float)
    // ========================================================================
    
    // Load kernel arguments
    s_load_dwordx2 s[4:5], s[0:1], 0      // output ptr
    s_load_dwordx2 s[6:7], s[0:1], 16     // Q ptr
    s_load_dwordx2 s[10:11], s[0:1], 32   // K ptr
    s_load_dwordx2 s[14:15], s[0:1], 48   // V ptr
    s_load_dword s29, s[0:1], 80          // scale
    s_load_dword s30, s[0:1], 96          // seqlen_k
    s_load_dword s26, s[0:1], 512         // q_scale
    s_load_dword s27, s[0:1], 516         // k_scale
    s_load_dword s28, s[0:1], 520         // v_scale
    
    s_waitcnt lgkmcnt(0)
    
    // Store output ptr in s8:s9 for later use
    s_mov_b32 s8, s4
    s_mov_b32 s9, s5
    
    // Calculate K loop count: ceil(seqlen_k / 32)
    s_add_u32 s30, s30, 31
    s_lshr_b32 s30, s30, 5
    
    // Thread ID
    v_mov_b32_e32 v0, 0
    v_mbcnt_lo_u32_b32 v0, -1, 0
    v_mbcnt_hi_u32_b32 v0, -1, v0
    
    // Thread offset: tid * 16 bytes (each thread loads 16 FP8 values)
    v_lshlrev_b32_e32 v6, 4, v0
    
    // ========================================================================
    // LOAD Q TILE TO LDS
    // ========================================================================
    v_mov_b32_e32 v2, s6
    v_mov_b32_e32 v3, s7
    v_add_co_u32_e32 v2, vcc, v6, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    
    flat_load_dwordx4 v[80:83], v[2:3]
    v_add_co_u32_e32 v2, vcc, 4096, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    flat_load_dwordx4 v[84:87], v[2:3]
    
    s_waitcnt vmcnt(0)
    
    // Store Q to LDS
    v_add_u32_e32 v7, LDS_Q_OFFSET, v6
    ds_write_b128 v7, v[80:83]
    v_add_u32_e32 v7, LDS_Q_OFFSET + 1024, v6
    ds_write_b128 v7, v[84:87]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // INITIALIZE ACCUMULATORS
    // ========================================================================
    s_mov_b32 s37, 0
    v_mov_b32_e32 v16, 0xff800000         // running_max = -inf
    v_mov_b32_e32 v17, s37                // running_sum = 0
    v_mov_b32_e32 v48, s37                // output accumulator v48-v63
    v_mov_b32_e32 v49, s37
    v_mov_b32_e32 v50, s37
    v_mov_b32_e32 v51, s37
    v_mov_b32_e32 v52, s37
    v_mov_b32_e32 v53, s37
    v_mov_b32_e32 v54, s37
    v_mov_b32_e32 v55, s37
    v_mov_b32_e32 v56, s37
    v_mov_b32_e32 v57, s37
    v_mov_b32_e32 v58, s37
    v_mov_b32_e32 v59, s37
    v_mov_b32_e32 v60, s37
    v_mov_b32_e32 v61, s37
    v_mov_b32_e32 v62, s37
    v_mov_b32_e32 v63, s37
    
    // K loop counter
    s_mov_b32 s31, 0
    s_mov_b32 s33, 0                      // k_offset
    
    // ========================================================================
    // K-LOOP START
    // ========================================================================
K_LOOP:
    // ========================================================================
    // LOAD K TILE TO LDS
    // ========================================================================
    v_mov_b32_e32 v2, s10
    v_mov_b32_e32 v3, s11
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v2, vcc, v7, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    v_add_co_u32_e32 v2, vcc, v6, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    
    flat_load_dwordx4 v[80:83], v[2:3]
    v_add_co_u32_e32 v2, vcc, 4096, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    flat_load_dwordx4 v[84:87], v[2:3]
    
    s_waitcnt vmcnt(0)
    
    // Store K to LDS
    v_add_u32_e32 v7, LDS_K_OFFSET, v6
    ds_write_b128 v7, v[80:83]
    v_add_u32_e32 v7, LDS_K_OFFSET + 1024, v6
    ds_write_b128 v7, v[84:87]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // QK MFMA (Q × K^T for head_dim=128)
    // ========================================================================
    // Initialize QK accumulator
    v_mov_b32_e32 v32, s37
    v_mov_b32_e32 v33, s37
    v_mov_b32_e32 v34, s37
    v_mov_b32_e32 v35, s37
    v_mov_b32_e32 v36, s37
    v_mov_b32_e32 v37, s37
    v_mov_b32_e32 v38, s37
    v_mov_b32_e32 v39, s37
    v_mov_b32_e32 v40, s37
    v_mov_b32_e32 v41, s37
    v_mov_b32_e32 v42, s37
    v_mov_b32_e32 v43, s37
    v_mov_b32_e32 v44, s37
    v_mov_b32_e32 v45, s37
    v_mov_b32_e32 v46, s37
    v_mov_b32_e32 v47, s37
    
    // Thread offset for MFMA operand loading
    v_and_b32_e32 v7, 31, v0              // lane within 32
    v_lshlrev_b32_e32 v7, 3, v7           // * 8 bytes
    
    // Load Q K=0..31 into AGPRs
    v_add_u32_e32 v9, 0, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 16, v7
    ds_read_b64 a[2:3], v9
    
    // Load K K=0..31
    v_add_u32_e32 v9, LDS_K_OFFSET, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=0..15, K=16..31
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=32..63
    v_add_u32_e32 v9, 32, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[2:3], v9
    
    // Load K K=32..63
    v_add_u32_e32 v9, LDS_K_OFFSET + 32, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=32..47, K=48..63
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=64..95
    v_add_u32_e32 v9, 64, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[2:3], v9
    
    // Load K K=64..95
    v_add_u32_e32 v9, LDS_K_OFFSET + 64, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=64..79, K=80..95
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=96..127
    v_add_u32_e32 v9, 96, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    
    // Load K K=96..127
    v_add_u32_e32 v9, LDS_K_OFFSET + 96, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=96..111, K=112..127
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    // ========================================================================
    // APPLY QK SCALE
    // ========================================================================
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
    
    // ========================================================================
    // COMPUTE LOCAL MAX (thread's 16 values -> single max)
    // ========================================================================
    v_max_f32_e32 v18, v32, v33
    v_max_f32_e32 v19, v34, v35
    v_max_f32_e32 v18, v18, v19
    v_max_f32_e32 v19, v36, v37
    v_max_f32_e32 v20, v38, v39
    v_max_f32_e32 v19, v19, v20
    v_max_f32_e32 v18, v18, v19
    v_max_f32_e32 v19, v40, v41
    v_max_f32_e32 v20, v42, v43
    v_max_f32_e32 v19, v19, v20
    v_max_f32_e32 v20, v44, v45
    v_max_f32_e32 v21, v46, v47
    v_max_f32_e32 v20, v20, v21
    v_max_f32_e32 v19, v19, v20
    v_max_f32_e32 v18, v18, v19           // v18 = local_max
    
    // Cross-thread max reduction
    v_mov_b32_e32 v19, v18
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v19, v18
    v_max_f32_e32 v18, v18, v19           // v18 = row max
    
    // Compute new_max and correction factor
    v_mov_b32_e32 v19, v16                // old_max
    v_max_f32_e32 v16, v16, v18           // running_max = max(running_max, row_max)
    v_sub_f32_e32 v20, v19, v16
    v_exp_f32_e32 v19, v20                // correction = exp(old_max - new_max)
    
    // Scale previous output by correction
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
    
    // Scale running_sum by correction
    v_mul_f32_e32 v17, v19, v17
    
    // ========================================================================
    // COMPUTE exp(QK - new_max) 
    // ========================================================================
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
    
    // ========================================================================
    // LOCAL SUM (sum P values)
    // ========================================================================
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
    v_add_f32_e32 v20, v20, v21           // v20 = local_sum
    
    // Cross-thread sum reduction
    v_mov_b32_e32 v21, v20
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v21, v20
    v_add_f32_e32 v20, v20, v21           // v20 = row sum
    
    // Update running sum
    v_add_f32_e32 v17, v17, v20
    
    // ========================================================================
    // LOAD V TILE AND COMPUTE P×V
    // ========================================================================
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[80:83], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[84:87], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store V to LDS
    v_add_u32_e32 v7, LDS_V_OFFSET, v6
    ds_write_b128 v7, v[80:83]
    v_add_u32_e32 v7, LDS_V_OFFSET + 4096, v6
    ds_write_b128 v7, v[84:87]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // PACK P VALUES TO FP8 FOR PV MFMA
    // ========================================================================
    v_cvt_pk_fp8_f32 v21, v32, v33
    v_cvt_pk_fp8_f32 v70, v34, v35
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v21, v21, v70
    
    v_cvt_pk_fp8_f32 v22, v36, v37
    v_cvt_pk_fp8_f32 v70, v38, v39
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v22, v22, v70
    
    v_cvt_pk_fp8_f32 v23, v40, v41
    v_cvt_pk_fp8_f32 v70, v42, v43
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v23, v23, v70
    
    v_cvt_pk_fp8_f32 v24, v44, v45
    v_cvt_pk_fp8_f32 v70, v46, v47
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v24, v24, v70
    
    // Move P to AGPRs
    v_accvgpr_write_b32 a0, v21
    v_accvgpr_write_b32 a1, v22
    v_accvgpr_write_b32 a2, v23
    v_accvgpr_write_b32 a3, v24
    
    // ========================================================================
    // PV MFMAs
    // ========================================================================
    v_and_b32_e32 v7, 31, v0
    v_lshlrev_b32_e32 v7, 3, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    
    ds_read_b64 v[64:65], v7
    ds_read_b64 v[66:67], v7 offset:128
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    ds_read_b64 v[64:65], v7 offset:256
    ds_read_b64 v[66:67], v7 offset:384
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[64:65], v[48:63]
    
    // Update k_offset for next K iteration
    s_add_u32 s33, s33, 1024
    
    // ========================================================================
    // K-LOOP INCREMENT
    // ========================================================================
    s_add_u32 s31, s31, 1
    s_cmp_lt_u32 s31, s30
    s_cbranch_scc1 K_LOOP
    
    // ========================================================================
    // FINAL NORMALIZATION
    // ========================================================================
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    
    v_rcp_f32_e32 v17, v17
    
    // Wait for rcp result - use ds_nop to create dependency
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    s_nop 0
    
    // Store v17 to LDS and read back to ensure it's ready
    v_lshlrev_b32_e32 v3, 2, v0
    ds_write_b32 v3, v17
    s_waitcnt lgkmcnt(0)
    ds_read_b32 v17, v3
    s_waitcnt lgkmcnt(0)
    
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
    
    // Apply V scale
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
    
    // ========================================================================
    // STORE OUTPUT
    // ========================================================================
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v3, 6, v0
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_store_dwordx4 v[10:11], v[48:51]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[52:55]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[56:59]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[60:63]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter18fmha_fwd_hd128_fp8E, .-_ZN5aiter18fmha_fwd_hd128_fp8E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter18fmha_fwd_hd128_fp8E
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

.section .note.GNU-stack

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter18fmha_fwd_hd128_fp8E
    .symbol: _ZN5aiter18fmha_fwd_hd128_fp8E.kd
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

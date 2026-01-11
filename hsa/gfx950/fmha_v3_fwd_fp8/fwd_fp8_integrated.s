// FP8 Flash Attention kernel for head_dim=32 - Full integration
// Following BF16 kernel architecture:
// 1. Q/K loaded directly from global
// 2. QK MFMA -> S in VGPRs v[32:47]
// 3. Softmax in VGPRs with cross-thread reduction using ds_swizzle
// 4. P stays in VGPRs
// 5. Output P (for testing; full version would continue with PV)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter17fwd_fp8_integratedE
.p2align 8
.type _ZN5aiter17fwd_fp8_integratedE,@function

_ZN5aiter17fwd_fp8_integratedE:
    s_mov_b64 exec, -1
    
    // Args: 0=O, 8=Q, 16=K, 24=V
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_O (F32)
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_Q (FP8)
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // ptr_K (FP8)
    s_load_dwordx2 s[14:15], s[0:1], 0x18  // ptr_V (FP8)
    
    v_and_b32_e32 v0, 63, v0               // tid
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // PHASE 1: QK MFMA -> S[32,32] in v[32:47]
    // ========================================================================
    
    // Initialize S accumulators
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
    
    // Thread mapping
    v_and_b32_e32 v1, 31, v0              // m = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // d_group = tid / 32
    v_lshlrev_b32_e32 v2, 3, v2           // d_offset = d_group * 8
    v_lshlrev_b32_e32 v3, 5, v1           // m * 32 (row stride)
    
    // QK Pass 0
    v_add_u32_e32 v4, v3, v2              // m*32 + d_offset
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[20:21], v[10:11]
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[22:23], v[10:11]
    
    s_waitcnt vmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // QK Pass 1
    v_add_u32_e32 v4, 16, v3
    v_add_u32_e32 v4, v4, v2
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[20:21], v[10:11]
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx2 v[22:23], v[10:11]
    
    s_waitcnt vmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    s_nop 15
    s_nop 15
    
    // ========================================================================
    // PHASE 2: Softmax S -> P in VGPRs
    // ========================================================================
    // ds_swizzle SWAP,N within 64 lanes actually stays within 32-lane halves:
    // - SWAP,16: lane i <-> lane i XOR 16 (stays within 0-31 or 32-63)
    // - This is correct for row-wise reduction!
    
    s_mov_b32 s2, 0x3fb8aa3b   // log2(e)
    
    // Macro-like pattern for each row register (unrolled)
    // For each v[32+r], we:
    // 1. Find max across 32 threads using butterfly reduction
    // 2. Compute exp((S - max) * log2(e))
    // 3. Sum across 32 threads
    // 4. Normalize: P = exp / sum

// ======================= ROW 0 (v32) =======================
    v_mov_b32_e32 v50, v32
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    // v50 = max for this row
    v_sub_f32_e32 v60, v32, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    // Reduce sum
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v32, v60, v54

// ======================= ROW 1 (v33) =======================
    v_mov_b32_e32 v50, v33
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v33, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v33, v60, v54

// ======================= ROW 2 (v34) =======================
    v_mov_b32_e32 v50, v34
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v34, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v34, v60, v54

// ======================= ROW 3 (v35) =======================
    v_mov_b32_e32 v50, v35
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v35, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v35, v60, v54

// ======================= ROW 4-15 (v36-v47) - Same pattern =======================
// ROW 4 (v36)
    v_mov_b32_e32 v50, v36
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v36, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v36, v60, v54

// ROW 5 (v37)
    v_mov_b32_e32 v50, v37
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v37, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v37, v60, v54

// ROW 6 (v38)
    v_mov_b32_e32 v50, v38
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v38, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v38, v60, v54

// ROW 7 (v39)
    v_mov_b32_e32 v50, v39
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v39, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v39, v60, v54

// ROW 8 (v40)
    v_mov_b32_e32 v50, v40
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v40, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v40, v60, v54

// ROW 9 (v41)
    v_mov_b32_e32 v50, v41
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v41, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v41, v60, v54

// ROW 10 (v42)
    v_mov_b32_e32 v50, v42
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v42, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v42, v60, v54

// ROW 11 (v43)
    v_mov_b32_e32 v50, v43
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v43, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v43, v60, v54

// ROW 12 (v44)
    v_mov_b32_e32 v50, v44
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v44, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v44, v60, v54

// ROW 13 (v45)
    v_mov_b32_e32 v50, v45
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v45, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v45, v60, v54

// ROW 14 (v46)
    v_mov_b32_e32 v50, v46
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v46, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v46, v60, v54

// ROW 15 (v47)
    v_mov_b32_e32 v50, v47
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    ds_swizzle_b32 v51, v50 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v50, v50, v51
    v_sub_f32_e32 v60, v47, v50
    v_mov_b32_e32 v61, s2
    v_mul_f32_e32 v60, v60, v61
    v_exp_f32_e32 v60, v60
    s_nop 7
    v_mov_b32_e32 v52, v60
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,16)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,8)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,4)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,2)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    ds_swizzle_b32 v53, v52 offset:swizzle(SWAP,1)
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v52, v52, v53
    v_rcp_f32_e32 v54, v52
    s_nop 7
    v_mul_f32_e32 v47, v60, v54

    // ========================================================================
    // PHASE 3: Write P to LDS for redistribution
    // ========================================================================
    // P is in v[32:47] with QK MFMA output layout (interleaved rows)
    // LDS layout: P[M, K] at offset M*128 + K*4 (32 cols * 4 bytes per row)
    
    v_and_b32_e32 v1, 31, v0              // K = tid % 32 (column being written)
    v_lshrrev_b32_e32 v2, 5, v0           // group = tid / 32
    v_lshlrev_b32_e32 v3, 2, v1           // K * 4 (column byte offset)
    v_lshlrev_b32_e32 v4, 9, v2           // group * 512 (= group * 4 rows * 128 bytes)
    v_add_u32_e32 v5, v3, v4              // base = K*4 + group*512
    
    // Write v32-v35 to rows M_base+0,1,2,3
    ds_write_b32 v5, v32                  // P[M_base+0, K]
    ds_write_b32 v5, v33 offset:128       // P[M_base+1, K]
    ds_write_b32 v5, v34 offset:256       // P[M_base+2, K]
    ds_write_b32 v5, v35 offset:384       // P[M_base+3, K]
    
    // Write v36-v39 to rows M_base+8,9,10,11
    ds_write_b32 v5, v36 offset:1024      // P[M_base+8, K]
    ds_write_b32 v5, v37 offset:1152      // P[M_base+9, K]
    ds_write_b32 v5, v38 offset:1280      // P[M_base+10, K]
    ds_write_b32 v5, v39 offset:1408      // P[M_base+11, K]
    
    // Write v40-v43 to rows M_base+16,17,18,19
    ds_write_b32 v5, v40 offset:2048      // P[M_base+16, K]
    ds_write_b32 v5, v41 offset:2176      // P[M_base+17, K]
    ds_write_b32 v5, v42 offset:2304      // P[M_base+18, K]
    ds_write_b32 v5, v43 offset:2432      // P[M_base+19, K]
    
    // Write v44-v47 to rows M_base+24,25,26,27
    ds_write_b32 v5, v44 offset:3072      // P[M_base+24, K]
    ds_write_b32 v5, v45 offset:3200      // P[M_base+25, K]
    ds_write_b32 v5, v46 offset:3328      // P[M_base+26, K]
    ds_write_b32 v5, v47 offset:3456      // P[M_base+27, K]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // PHASE 4: Read P in PV MFMA layout (transposed)
    // ========================================================================
    // PV MFMA A operand: Thread t needs P[M=t%32, K_start:K_start+8]
    // K_start = (t/32) * 8
    
    v_and_b32_e32 v6, 31, v0              // M = tid % 32
    v_lshrrev_b32_e32 v7, 5, v0           // group = tid / 32
    v_lshlrev_b32_e32 v7, 5, v7           // K_start * 4 = group * 32
    v_lshlrev_b32_e32 v6, 7, v6           // M * 128
    v_add_u32_e32 v6, v6, v7              // offset = M*128 + K_start*4
    
    // Read 8 consecutive K values as F32
    ds_read_b128 v[64:67], v6             // P[M, K_start:K_start+4]
    ds_read_b128 v[68:71], v6 offset:16   // P[M, K_start+4:K_start+8]
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // PHASE 5: Convert P from F32 to FP8
    // ========================================================================
    // v_cvt_pk_fp8_f32 packs 2 F32 values into 2 FP8 bytes (low 16 bits)
    // NOTE: High 16 bits contain garbage - must mask before combining!
    
    v_cvt_pk_fp8_f32 v72, v64, v65        // [fp8(P0), fp8(P1)]
    v_and_b32_e32 v72, 0xFFFF, v72        // Mask to low 16 bits
    v_cvt_pk_fp8_f32 v73, v66, v67        // [fp8(P2), fp8(P3)]
    v_lshlrev_b32_e32 v73, 16, v73        // Shift clears low bits, garbage in low 16
    v_or_b32_e32 v20, v72, v73            // P_fp8[0:4] in v20
    
    v_cvt_pk_fp8_f32 v74, v68, v69        // [fp8(P4), fp8(P5)]
    v_and_b32_e32 v74, 0xFFFF, v74        // Mask to low 16 bits
    v_cvt_pk_fp8_f32 v75, v70, v71        // [fp8(P6), fp8(P7)]
    v_lshlrev_b32_e32 v75, 16, v75
    v_or_b32_e32 v21, v74, v75            // P_fp8[4:8] in v21
    
    // ========================================================================
    // PHASE 6: Load V from global memory
    // ========================================================================
    // V[32, 32] FP8 row-major: V[k, d] at offset k*32 + d
    // Thread t needs V[K_start:K_start+8, D=t%32] (8 values with stride 32)
    
    v_and_b32_e32 v1, 31, v0              // D = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // group = tid / 32
    v_lshlrev_b32_e32 v2, 8, v2           // K_start * 32 = group * 8 * 32 = group * 256
    
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_add_co_u32_e32 v10, vcc, v1, v10    // base + D
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v2, v10    // + K_start * 32
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 V values (one per K row, stride 32 bytes)
    flat_load_ubyte v76, v[10:11]
    flat_load_ubyte v77, v[10:11] offset:32
    flat_load_ubyte v78, v[10:11] offset:64
    flat_load_ubyte v79, v[10:11] offset:96
    flat_load_ubyte v80, v[10:11] offset:128
    flat_load_ubyte v81, v[10:11] offset:160
    flat_load_ubyte v82, v[10:11] offset:192
    flat_load_ubyte v83, v[10:11] offset:224
    
    s_waitcnt vmcnt(0)
    
    // Pack V bytes into 2 DWORDs for MFMA B operand
    v_lshlrev_b32_e32 v77, 8, v77
    v_or_b32_e32 v76, v76, v77
    v_lshlrev_b32_e32 v78, 16, v78
    v_or_b32_e32 v76, v76, v78
    v_lshlrev_b32_e32 v79, 24, v79
    v_or_b32_e32 v22, v76, v79            // V_fp8[0:4] in v22
    
    v_lshlrev_b32_e32 v81, 8, v81
    v_or_b32_e32 v80, v80, v81
    v_lshlrev_b32_e32 v82, 16, v82
    v_or_b32_e32 v80, v80, v82
    v_lshlrev_b32_e32 v83, 24, v83
    v_or_b32_e32 v23, v80, v83            // V_fp8[4:8] in v23
    
    // ========================================================================
    // PHASE 7: PV MFMA Pass 1 (K=0..15)
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
    
    // A operand: P_fp8 in v[20:21]
    // B operand: V_fp8 in v[22:23]
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========================================================================
    // PHASE 8: PV MFMA Pass 2 (K=16..31)
    // ========================================================================
    // Need to read P and V for second half of K dimension
    
    // Read P[M, K=16..23] or P[M, K=24..31] depending on group
    v_and_b32_e32 v6, 31, v0              // M = tid % 32
    v_lshrrev_b32_e32 v7, 5, v0           // group
    v_lshlrev_b32_e32 v7, 5, v7           // group * 32
    v_add_u32_e32 v7, 64, v7              // + 16*4 = +64 for second K half
    v_lshlrev_b32_e32 v6, 7, v6           // M * 128
    v_add_u32_e32 v6, v6, v7
    
    ds_read_b128 v[64:67], v6
    ds_read_b128 v[68:71], v6 offset:16
    
    // Load V for K=16..23 or K=24..31
    v_and_b32_e32 v1, 31, v0
    v_lshrrev_b32_e32 v2, 5, v0
    v_lshlrev_b32_e32 v2, 8, v2           // group * 256
    v_add_u32_e32 v2, 512, v2             // + 16*32 = +512 for second K half
    
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v2, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_ubyte v76, v[10:11]
    flat_load_ubyte v77, v[10:11] offset:32
    flat_load_ubyte v78, v[10:11] offset:64
    flat_load_ubyte v79, v[10:11] offset:96
    flat_load_ubyte v80, v[10:11] offset:128
    flat_load_ubyte v81, v[10:11] offset:160
    flat_load_ubyte v82, v[10:11] offset:192
    flat_load_ubyte v83, v[10:11] offset:224
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    
    // Convert P to FP8 (mask high bits - they contain garbage!)
    v_cvt_pk_fp8_f32 v72, v64, v65
    v_and_b32_e32 v72, 0xFFFF, v72
    v_cvt_pk_fp8_f32 v73, v66, v67
    v_lshlrev_b32_e32 v73, 16, v73
    v_or_b32_e32 v20, v72, v73
    
    v_cvt_pk_fp8_f32 v74, v68, v69
    v_and_b32_e32 v74, 0xFFFF, v74
    v_cvt_pk_fp8_f32 v75, v70, v71
    v_lshlrev_b32_e32 v75, 16, v75
    v_or_b32_e32 v21, v74, v75
    
    // Pack V
    v_lshlrev_b32_e32 v77, 8, v77
    v_or_b32_e32 v76, v76, v77
    v_lshlrev_b32_e32 v78, 16, v78
    v_or_b32_e32 v76, v76, v78
    v_lshlrev_b32_e32 v79, 24, v79
    v_or_b32_e32 v22, v76, v79
    
    v_lshlrev_b32_e32 v81, 8, v81
    v_or_b32_e32 v80, v80, v81
    v_lshlrev_b32_e32 v82, 16, v82
    v_or_b32_e32 v80, v80, v82
    v_lshlrev_b32_e32 v83, 24, v83
    v_or_b32_e32 v23, v80, v83
    
    // MFMA Pass 2
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    s_nop 15
    s_nop 7
    
    // ========================================================================
    // PHASE 9: Store O to global memory
    // ========================================================================
    // Same scatter pattern as before (interleaved rows)
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0              // N = tid % 32
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4 bytes
    v_lshrrev_b32_e32 v4, 5, v0
    v_lshlrev_b32_e32 v4, 2, v4           // M_base = (tid/32) * 4
    
    // Store rows M_base + 0,1,2,3
    v_lshlrev_b32_e32 v5, 7, v4           // M_base * 128
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v32
    flat_store_dword v[12:13], v33 offset:128
    flat_store_dword v[12:13], v34 offset:256
    flat_store_dword v[12:13], v35 offset:384
    
    // Store rows M_base + 8,9,10,11
    v_add_u32_e32 v6, 8, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v36
    flat_store_dword v[12:13], v37 offset:128
    flat_store_dword v[12:13], v38 offset:256
    flat_store_dword v[12:13], v39 offset:384
    
    // Store rows M_base + 16,17,18,19
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v40
    flat_store_dword v[12:13], v41 offset:128
    flat_store_dword v[12:13], v42 offset:256
    flat_store_dword v[12:13], v43 offset:384
    
    // Store rows M_base + 24,25,26,27
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

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter17fwd_fp8_integratedE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 88
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 84
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter17fwd_fp8_integratedE
    .symbol: _ZN5aiter17fwd_fp8_integratedE.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 88
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
      - .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
      - .size: 8
        .offset: 24
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

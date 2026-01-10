// SPDX-License-Identifier: MIT
// FP8 Flash Attention - V×P version 2 with proper P redistribution
// Key insight: After QK MFMA, thread t has P[Q_range, K=t%32]
//              For V×P B operand, need P[Q=t%32, K_range]
//              Must write P to LDS and re-read with correct pattern

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E
.p2align 8
.type _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E,@function

.set BLOCK_M, 64
.set BLOCK_N, 64
.set HEAD_DIM, 128
.set THREADS, 256
.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 8192
.set LDS_V_OFFSET, 16384
.set LDS_P_OFFSET, 24576           // P scratch area: 32×32 FP8 = 1KB

_ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E:
    // Load kernel arguments
    s_and_b32 s1, s1, 0xffff
    
    s_load_dwordx2 s[8:9], s[0:1], 0x00       // ptr_R
    s_load_dwordx2 s[10:11], s[0:1], 0x10     // ptr_Q
    s_load_dwordx2 s[12:13], s[0:1], 0x20     // ptr_K
    s_load_dwordx2 s[14:15], s[0:1], 0x30     // ptr_V
    s_load_dword s20, s[0:1], 0x50            // softmax_scale
    s_load_dword s21, s[0:1], 0x58            // seqlen_q
    s_load_dword s22, s[0:1], 0x60            // seqlen_k
    s_load_dword s26, s[0:1], 0x200           // q_scale
    s_load_dword s27, s[0:1], 0x204           // k_scale
    s_load_dword s28, s[0:1], 0x208           // v_scale
    
    v_and_b32_e32 v0, 0xff, v0                // tid (0-255)
    v_lshrrev_b32_e32 v1, 6, v0               // warp_id (0-3)
    v_and_b32_e32 v2, 63, v0                  // lane_id (0-63)
    
    s_waitcnt lgkmcnt(0)
    
    // Compute scales and offsets
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
    
    // Initialize online softmax
    v_mov_b32_e32 v16, 0xff800000             // running_max = -inf
    v_mov_b32_e32 v17, 0                      // running_sum = 0
    
    // Initialize output accumulators
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
    
    // K-tile loop
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
    
    // Initialize QK accumulator
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

    // Apply QK scale
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
    
    // Online softmax: find max
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

    // Scale previous accumulators
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
    
    // Compute exp(QK - max) = P
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
    
    // Sum P for running_sum
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
    // WRITE P TO LDS FOR REDISTRIBUTION
    // After QK MFMA, thread t owns P[Q_base:Q_base+16, K=t%32]
    // where Q_base = (t/32) * 16
    // 
    // Store layout: P[Q_row, K_col] at LDS_P_OFFSET + Q_row * 32 + K_col
    // Thread t stores its 16 P values at column K=t%32
    // ========================================================================
    
    // Convert P to FP8 for storage
    v_mov_b32_e32 v80, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v80, v32, v33        // P[0:1]
    v_cvt_pk_fp8_f32 v81, v34, v35        // P[2:3]
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v80, v80, v81            // v80 = P[0:3]
    
    v_mov_b32_e32 v82, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v82, v36, v37
    v_cvt_pk_fp8_f32 v81, v38, v39
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v82, v82, v81            // v82 = P[4:7]
    
    v_mov_b32_e32 v84, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v84, v40, v41
    v_cvt_pk_fp8_f32 v81, v42, v43
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v84, v84, v81            // v84 = P[8:11]
    
    v_mov_b32_e32 v86, 0
    v_mov_b32_e32 v81, 0
    v_cvt_pk_fp8_f32 v86, v44, v45
    v_cvt_pk_fp8_f32 v81, v46, v47
    v_lshlrev_b32_e32 v81, 16, v81
    v_or_b32_e32 v86, v86, v81            // v86 = P[12:15]
    
    // LDS address for P: Q_base * 32 + K_col
    // Q_base = (lane_id / 32) * 16 = (lane_id / 32) * 16
    // K_col = lane_id % 32
    v_lshrrev_b32_e32 v7, 5, v2           // lane_id / 32 (0 or 1)
    v_lshlrev_b32_e32 v7, 9, v7           // (lane_id/32) * 16 * 32 = (lane_id/32) * 512
    v_and_b32_e32 v8, 31, v2              // K_col = lane_id % 32
    v_add_u32_e32 v7, v7, v8              // Q_base * 32 + K_col
    v_add_u32_e32 v7, LDS_P_OFFSET, v7    // + LDS base
    
    // Store P values row by row (4 rows per dword, stride = 32 bytes)
    // v80 = P[Q_base+0..3, K], store at rows Q_base+0,1,2,3
    ds_write_b8 v7, v80                   // P[Q_base+0, K]
    v_lshrrev_b32_e32 v81, 8, v80
    v_add_u32_e32 v9, 32, v7
    ds_write_b8 v9, v81                   // P[Q_base+1, K]
    v_lshrrev_b32_e32 v81, 16, v80
    v_add_u32_e32 v9, 64, v7
    ds_write_b8 v9, v81                   // P[Q_base+2, K]
    v_lshrrev_b32_e32 v81, 24, v80
    v_add_u32_e32 v9, 96, v7
    ds_write_b8 v9, v81                   // P[Q_base+3, K]
    
    // v82 = P[Q_base+4..7, K]
    ds_write_b8_d16_hi v7, v82 offset:128 // Actually, need to unpack properly
    // Let me redo this...
    v_add_u32_e32 v9, 128, v7
    ds_write_b8 v9, v82                   // P[Q_base+4, K]
    v_lshrrev_b32_e32 v81, 8, v82
    v_add_u32_e32 v9, 160, v7
    ds_write_b8 v9, v81                   // P[Q_base+5, K]
    v_lshrrev_b32_e32 v81, 16, v82
    v_add_u32_e32 v9, 192, v7
    ds_write_b8 v9, v81                   // P[Q_base+6, K]
    v_lshrrev_b32_e32 v81, 24, v82
    v_add_u32_e32 v9, 224, v7
    ds_write_b8 v9, v81                   // P[Q_base+7, K]
    
    // v84 = P[Q_base+8..11, K]
    v_add_u32_e32 v9, 256, v7
    ds_write_b8 v9, v84
    v_lshrrev_b32_e32 v81, 8, v84
    v_add_u32_e32 v9, 288, v7
    ds_write_b8 v9, v81
    v_lshrrev_b32_e32 v81, 16, v84
    v_add_u32_e32 v9, 320, v7
    ds_write_b8 v9, v81
    v_lshrrev_b32_e32 v81, 24, v84
    v_add_u32_e32 v9, 352, v7
    ds_write_b8 v9, v81
    
    // v86 = P[Q_base+12..15, K]
    v_add_u32_e32 v9, 384, v7
    ds_write_b8 v9, v86
    v_lshrrev_b32_e32 v81, 8, v86
    v_add_u32_e32 v9, 416, v7
    ds_write_b8 v9, v81
    v_lshrrev_b32_e32 v81, 16, v86
    v_add_u32_e32 v9, 448, v7
    ds_write_b8 v9, v81
    v_lshrrev_b32_e32 v81, 24, v86
    v_add_u32_e32 v9, 480, v7
    ds_write_b8 v9, v81
    
    s_barrier
    
    // ========================================================================
    // READ P FOR V×P B OPERAND
    // B operand layout: thread t needs P[Q=t%32, K_range] where
    // K_range = (t/32)*8 : (t/32)*8+8
    // 
    // LDS layout: P[Q, K] at Q*32 + K
    // Read address: LDS_P_OFFSET + (t%32)*32 + (t/32)*8
    // Need to read 8 bytes (8 K values)
    // ========================================================================
    
    v_and_b32_e32 v7, 31, v2              // Q = lane_id % 32
    v_lshlrev_b32_e32 v7, 5, v7           // Q * 32
    v_lshrrev_b32_e32 v8, 5, v2           // K_group = lane_id / 32
    v_lshlrev_b32_e32 v8, 3, v8           // K_group * 8
    v_add_u32_e32 v7, v7, v8              // Q*32 + K_group*8
    v_add_u32_e32 v7, LDS_P_OFFSET, v7
    
    // Read 8 FP8 values into 2 dwords for MFMA B operand
    ds_read_b64 v[82:83], v7              // P[Q=t%32, K=(t/32)*8..(t/32)*8+7]
    
    // Load V for MFMA A operand
    v_and_b32_e32 v80, 31, v0             // d = tid % 32 (D position)
    v_lshrrev_b32_e32 v81, 5, v0          // k_group = tid / 32
    
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v80, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // K_start = k_group * 8, offset = K_start * 128 = k_group * 1024
    v_lshlrev_b32_e32 v7, 10, v81
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 K values for V MFMA 1
    flat_load_ubyte v72, v[10:11]
    flat_load_ubyte v73, v[10:11] offset:128
    flat_load_ubyte v74, v[10:11] offset:256
    flat_load_ubyte v75, v[10:11] offset:384
    flat_load_ubyte v76, v[10:11] offset:512
    flat_load_ubyte v77, v[10:11] offset:640
    flat_load_ubyte v78, v[10:11] offset:768
    flat_load_ubyte v79, v[10:11] offset:896
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    
    // Pack V into 2 dwords for AGPRs
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
    
    v_accvgpr_write_b32 a0, v64
    v_accvgpr_write_b32 a1, v65
    
    s_nop 7

    // V×P MFMA 1: V[K=0..15] × P[K=0..15]
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[82:83], v[48:63]
    
    // Load V for K=16..31 and P for second half
    // V: K_start = 16 + k_group*8
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v80, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, 2048, v10  // K=16 base
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_lshlrev_b32_e32 v7, 10, v81
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_ubyte v72, v[10:11]
    flat_load_ubyte v73, v[10:11] offset:128
    flat_load_ubyte v74, v[10:11] offset:256
    flat_load_ubyte v75, v[10:11] offset:384
    flat_load_ubyte v76, v[10:11] offset:512
    flat_load_ubyte v77, v[10:11] offset:640
    flat_load_ubyte v78, v[10:11] offset:768
    flat_load_ubyte v79, v[10:11] offset:896
    
    // Read P for K=16..31: LDS_P_OFFSET + Q*32 + 16 + k_group*8
    v_and_b32_e32 v7, 31, v2
    v_lshlrev_b32_e32 v7, 5, v7
    v_lshrrev_b32_e32 v8, 5, v2
    v_lshlrev_b32_e32 v8, 3, v8
    v_add_u32_e32 v7, v7, v8
    v_add_u32_e32 v7, 16, v7              // + 16 for second K half
    v_add_u32_e32 v7, LDS_P_OFFSET, v7
    ds_read_b64 v[84:85], v7
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    
    // Pack V
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
    
    v_accvgpr_write_b32 a2, v66
    v_accvgpr_write_b32 a3, v67
    
    s_nop 7

    // V×P MFMA 2: V[K=16..31] × P[K=16..31]
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[84:85], v[48:63]
    
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
    
    s_nop 7

    // ========================================================================
    // STORE OUTPUT (TRANSPOSED)
    // V×P output: O^T[D, Q] where thread t owns O^T[D_base:D_base+16, Q=t%32]
    // Need O[Q, D] = O[Q=t%32, D_base:D_base+16]
    // Address: output_ptr + (t%32)*head_dim*sizeof(float) + (t/32)*16*sizeof(float)
    //        = output_ptr + (t%32)*512 + (t/32)*64
    // ========================================================================
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0
    v_lshrrev_b32_e32 v4, 5, v0
    v_lshlrev_b32_e32 v3, 9, v3           // Q * 512
    v_lshlrev_b32_e32 v4, 6, v4           // D_group * 64
    v_add_u32_e32 v3, v3, v4
    
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

.size _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E, .-_ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E
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
  - .name: _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E
    .symbol: _ZN5aiter25fmha_fwd_hd128_fp8_vxp_v2E.kd
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

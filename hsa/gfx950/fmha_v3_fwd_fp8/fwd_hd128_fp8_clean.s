// SPDX-License-Identifier: MIT
// FP8 Flash Attention - Clean version without debug stores
// Optimized for benchmarking

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter18fmha_fwd_hd128_fp8E
.p2align 8
.type _ZN5aiter18fmha_fwd_hd128_fp8E,@function

// Configuration constants
.set BLOCK_M, 64                // Q tile rows
.set BLOCK_N, 64                // K/V tile rows  
.set HEAD_DIM, 128              // Head dimension
.set THREADS, 256               // Threads per block (4 warps)
.set LDS_Q_SIZE, 8192           // 64 * 128 FP8 = 8KB for Q
.set LDS_K_SIZE, 8192           // 64 * 128 FP8 = 8KB for K
.set LDS_V_SIZE, 8192           // 64 * 128 FP8 = 8KB for V
.set LDS_TOTAL, 32768           // 32KB total LDS
.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 8192
.set LDS_V_OFFSET, 16384
.set Q_TILE_SIZE, 8192          // 64 * 128 bytes per Q-tile

_ZN5aiter18fmha_fwd_hd128_fp8E:
    // ========================================================================
    // REGISTER ALLOCATION:
    // s0-s1:   kernarg base pointer
    // s8-s9:   ptr_R (output)
    // s10-s11: ptr_Q
    // s12-s13: ptr_K
    // s14-s15: ptr_V
    // s20:     softmax_scale
    // s21:     seqlen_q
    // s22:     seqlen_k
    // s26:     q_scale
    // s27:     k_scale
    // s28:     v_scale
    // s30:     k_tiles (number of K-tiles to process)
    // s31:     current k_tile index
    // s32-s33: offsets
    // ========================================================================
    
    // Load kernel arguments
    s_and_b32 s1, s1, 0xffff
    
    s_load_dwordx2 s[8:9], s[0:1], 0x00       // ptr_R (output)
    s_load_dwordx2 s[10:11], s[0:1], 0x10     // ptr_Q
    s_load_dwordx2 s[12:13], s[0:1], 0x20     // ptr_K
    s_load_dwordx2 s[14:15], s[0:1], 0x30     // ptr_V
    s_load_dword s20, s[0:1], 0x50            // softmax_scale
    s_load_dword s21, s[0:1], 0x58            // seqlen_q
    s_load_dword s22, s[0:1], 0x60            // seqlen_k
    s_load_dword s26, s[0:1], 0x200           // q_scale
    s_load_dword s27, s[0:1], 0x204           // k_scale
    s_load_dword s28, s[0:1], 0x208           // v_scale
    
    // Thread/warp ID extraction
    v_and_b32_e32 v0, 0xff, v0                // tid (0-255)
    v_lshrrev_b32_e32 v1, 6, v0               // warp_id (0-3)
    v_and_b32_e32 v2, 63, v0                  // lane_id (0-63)
    
    s_waitcnt lgkmcnt(0)
    
    // head_stride = seqlen * 128
    s_lshl_b32 s23, s21, 7                    // s23 = seqlen_q * 128
    s_lshl_b32 s24, s22, 7                    // s24 = seqlen_k * 128
    
    // Combined QK scale = softmax_scale * q_scale * k_scale
    // For now, just use softmax_scale
    s_mov_b32 s29, s20
    
    // Simple offsets (single tile for now)
    s_mov_b32 s32, 0                          // q_offset = 0
    s_mov_b32 s30, 1                          // k_tiles = 1
    
    // ========================================================================
    // INITIALIZE OUTPUT ACCUMULATORS AND SOFTMAX STATE
    // ========================================================================
    // v16 = running_max (init to -inf)
    // v17 = running_sum (init to 0)
    // v48-v63 = output accumulator (init to 0)
    v_mov_b32_e32 v16, 0xff800000             // -inf
    v_mov_b32_e32 v17, 0                      // sum = 0
    
    // Initialize output accumulator to 0
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
    
    // ========================================================================
    // LOAD Q TILE TO LDS
    // ========================================================================
    v_lshlrev_b32_e32 v6, 4, v0               // thread offset = tid * 16
    
    // Load first 4KB of Q
    v_mov_b32_e32 v10, s10                    // Q base low
    v_mov_b32_e32 v11, s11                    // Q base high
    v_add_co_u32_e32 v10, vcc, v6, v10        // add thread offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[64:67], v[10:11]
    
    // Load second 4KB of Q
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[68:71], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store Q to LDS
    ds_write_b128 v6, v[64:67]
    v_add_u32_e32 v7, 4096, v6
    ds_write_b128 v7, v[68:71]
    
    s_barrier
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
    s_mov_b32 s31, 0                          // k_tile_idx = 0
    
K_LOOP:
    // ========================================================================
    // LOAD K TILE TO LDS
    // ========================================================================
    s_lshl_b32 s33, s31, 13                   // k_offset = k_tile_idx * 8192
    v_mov_b32_e32 v10, s12                    // K base low
    v_mov_b32_e32 v11, s13                    // K base high
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
    
    // Store K to LDS
    v_add_u32_e32 v7, LDS_K_OFFSET, v6
    ds_write_b128 v7, v[72:75]
    v_add_u32_e32 v7, LDS_K_OFFSET + 4096, v6
    ds_write_b128 v7, v[76:79]
    
    s_barrier
    
    // ========================================================================
    // QK MFMA - Initialize accumulator
    // ========================================================================
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
    
    // LDS read offset for MFMA
    v_lshlrev_b32_e32 v7, 3, v2           // v7 = lane_id * 8
    
    // ========================================================================
    // QK MFMAs - 8 iterations for HEAD_DIM=128
    // ========================================================================
    // K=0..31
    ds_read_b64 a[0:1], v7
    v_add_u32_e32 v9, 16, v7
    ds_read_b64 a[2:3], v9
    v_add_u32_e32 v9, LDS_K_OFFSET, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 16, v7
    ds_read_b64 v[66:67], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // K=32..63
    v_add_u32_e32 v9, 32, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 48, v7
    ds_read_b64 a[6:7], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 32, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 48, v7
    ds_read_b64 v[70:71], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    // K=64..95
    v_add_u32_e32 v9, 64, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 80, v7
    ds_read_b64 a[2:3], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 64, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 80, v7
    ds_read_b64 v[66:67], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // K=96..127
    v_add_u32_e32 v9, 96, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 112, v7
    ds_read_b64 a[6:7], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 96, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, LDS_K_OFFSET + 112, v7
    ds_read_b64 v[70:71], v9
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    // ========================================================================
    // ONLINE SOFTMAX
    // ========================================================================
    // Apply scale: QK = QK * softmax_scale
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
    
    // Local max reduction
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
    
    // Cross-thread max reduction
    v_mov_b32_e32 v19, v18
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v19, v18
    v_max_f32_e32 v18, v18, v19
    
    // Update running max
    v_mov_b32_e32 v19, v16               // old_max
    v_max_f32_e32 v16, v16, v18          // new running_max
    
    // Correction factor
    v_sub_f32_e32 v20, v19, v16
    v_exp_f32_e32 v19, v20               // correction = exp(old_max - new_max)
    
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
    
    // Scale running sum
    v_mul_f32_e32 v17, v19, v17
    
    // Compute exp(QK - max)
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
    
    // Local sum
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
    
    // Cross-thread sum reduction
    v_mov_b32_e32 v21, v20
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v21, v20
    v_add_f32_e32 v20, v20, v21
    
    // Update running sum
    v_add_f32_e32 v17, v17, v20
    
    // ========================================================================
    // LOAD V AND COMPUTE P×V
    // ========================================================================
    v_mov_b32_e32 v10, s14                // V base low
    v_mov_b32_e32 v11, s15                // V base high
    v_mov_b32_e32 v7, s33                 // k_offset
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[72:75], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[76:79], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store V to LDS
    v_add_u32_e32 v7, LDS_V_OFFSET, v6
    ds_write_b128 v7, v[72:75]
    v_add_u32_e32 v7, LDS_V_OFFSET + 4096, v6
    ds_write_b128 v7, v[76:79]
    
    s_barrier
    
    // Pack P to FP8
    v_mov_b32_e32 v21, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v21, v32, v33
    v_cvt_pk_fp8_f32 v70, v34, v35
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v21, v21, v70
    
    v_mov_b32_e32 v22, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v22, v36, v37
    v_cvt_pk_fp8_f32 v70, v38, v39
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v22, v22, v70
    
    v_accvgpr_write_b32 a0, v21
    v_accvgpr_write_b32 a1, v22
    
    v_mov_b32_e32 v23, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v23, v40, v41
    v_cvt_pk_fp8_f32 v70, v42, v43
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v23, v23, v70
    
    v_mov_b32_e32 v24, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v24, v44, v45
    v_cvt_pk_fp8_f32 v70, v46, v47
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v24, v24, v70
    
    v_accvgpr_write_b32 a2, v23
    v_accvgpr_write_b32 a3, v24
    
    // Load V from LDS for PV MFMA
    v_add_u32_e32 v7, LDS_V_OFFSET, v6
    ds_read_b64 v[64:65], v7
    ds_read_b64 v[66:67], v7 offset:128
    
    s_waitcnt lgkmcnt(0)
    
    // PV MFMAs
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    ds_read_b64 v[64:65], v7 offset:256
    ds_read_b64 v[66:67], v7 offset:384
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[64:65], v[48:63]
    
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
    v_rcp_f32_e32 v17, v17               // 1/sum
    // Need to wait for rcp result - use readfirstlane to force completion
    s_nop 0
    s_nop 0
    v_readfirstlane_b32 s36, v17         // Force v17 to be ready
    s_nop 0
    s_nop 0
    
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
    v_lshlrev_b32_e32 v3, 6, v0           // offset = tid * 64
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

// ========================================================================
// KERNEL DESCRIPTOR
// ========================================================================
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
    .amdhsa_next_free_vgpr 88
    .amdhsa_next_free_sgpr 48
    .amdhsa_accum_offset 80
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

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
    .vgpr_count: 88
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

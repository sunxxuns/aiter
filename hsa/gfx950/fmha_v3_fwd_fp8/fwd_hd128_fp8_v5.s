// SPDX-License-Identifier: MIT
// FP8 Flash Attention - 4-Wave Multi-Warp Implementation
// Each wave processes 32 Q rows, total 128 Q rows per workgroup

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter18fmha_fwd_hd128_fp8E
.p2align 8
.type _ZN5aiter18fmha_fwd_hd128_fp8E,@function

// Configuration constants
.set BLOCK_M, 128               // Q tile rows (32 per wave × 4 waves)
.set BLOCK_N, 64                // K/V tile rows  
.set HEAD_DIM, 128              // Head dimension
.set THREADS, 256               // Threads per block (4 warps)
.set WAVES, 4                   // Number of waves
.set Q_ROWS_PER_WAVE, 32        // Q rows per wave (one MFMA M-tile)
.set LDS_Q_SIZE, 16384          // 128 * 128 FP8 = 16KB for Q
.set LDS_K_SIZE, 8192           // 64 * 128 FP8 = 8KB for K
.set LDS_V_SIZE, 4096           // 32 * 128 FP8 = 4KB for V (per D-tile)
.set LDS_TOTAL, 32768           // 32KB total LDS
.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 16384
.set LDS_V_OFFSET, 24576
.set Q_TILE_SIZE, 16384         // 128 * 128 bytes per Q-tile

_ZN5aiter18fmha_fwd_hd128_fp8E:
    // ========================================================================
    // WORKGROUP LAYOUT:
    // workgroup_id_x = Q-tile index (which 128-row block of Q)
    // workgroup_id_y = head index (which attention head)
    // workgroup_id_z = batch index
    //
    // THREAD LAYOUT (256 threads = 4 waves):
    // wave_id = tid >> 6 (0-3)
    // lane_id = tid & 63 (0-63)
    //
    // Each wave processes Q[wave_id*32 : (wave_id+1)*32-1, :]
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
    
    // Thread/wave ID extraction
    v_and_b32_e32 v0, 0xff, v0                // tid (0-255)
    v_lshrrev_b32_e32 v1, 6, v0               // wave_id (0-3)
    v_and_b32_e32 v2, 63, v0                  // lane_id (0-63)
    
    // Broadcast wave_id to SGPR for conditionals
    v_readfirstlane_b32 s5, v1                // s5 = wave_id (uniform within wave)
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // COMPUTE WORKGROUP OFFSETS
    // ========================================================================
    s_lshl_b32 s23, s21, 7                    // s23 = seqlen_q * 128 = head_stride_q
    s_lshl_b32 s24, s22, 7                    // s24 = seqlen_k * 128 = head_stride_kv
    
    // Q-tile offset = qtile_idx * BLOCK_M * HEAD_DIM = qtile_idx * 16384
    s_lshl_b32 s32, s2, 14                    // s32 = qtile_idx * 16384
    
    // ========================================================================
    // COMPUTE QK SCALE
    // ========================================================================
    v_mov_b32_e32 v3, s20
    v_mov_b32_e32 v4, s26
    v_mov_b32_e32 v5, s27
    v_mul_f32_e32 v3, v3, v4
    v_mul_f32_e32 v3, v3, v5
    v_readfirstlane_b32 s29, v3               // Combined scale in s29
    
    // k_tiles = (seqlen_k + BLOCK_N - 1) / BLOCK_N
    s_add_u32 s30, s22, 63
    s_lshr_b32 s30, s30, 6                    // s30 = k_tiles
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE (per-wave)
    // ========================================================================
    v_mov_b32_e32 v16, 0xff800000             // running_max = -inf
    v_mov_b32_e32 v17, 0                      // running_sum = 0
    
    // ========================================================================
    // COOPERATIVE Q LOADING (all 4 waves contribute)
    // 16KB Q data / 256 threads / 16 bytes per load = 4 iterations
    // ========================================================================
    v_lshlrev_b32_e32 v6, 4, v0               // thread offset = tid * 16
    
    // Q base with qtile offset
    v_mov_b32_e32 v10, s10                    // Q base low
    v_mov_b32_e32 v11, s11                    // Q base high
    v_mov_b32_e32 v8, s32                     // qtile offset
    v_add_co_u32_e32 v10, vcc, v8, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10        // + thread offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 4 chunks of 4KB each (256 threads × 16 bytes = 4KB)
    flat_load_dwordx4 v[64:67], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[68:71], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[72:75], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[76:79], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store Q to LDS (cooperative - all threads write different locations)
    ds_write_b128 v6, v[64:67]                // tid * 16
    v_add_u32_e32 v7, 4096, v6
    ds_write_b128 v7, v[68:71]                // tid * 16 + 4096
    v_add_u32_e32 v7, 4096, v7
    ds_write_b128 v7, v[72:75]                // tid * 16 + 8192
    v_add_u32_e32 v7, 4096, v7
    ds_write_b128 v7, v[76:79]                // tid * 16 + 12288
    
    s_barrier
    
    // ========================================================================
    // WAVE-SPECIFIC LDS OFFSET FOR Q ACCESS
    // Wave n reads from LDS[n * 32 * 128] = LDS[n * 4096]
    // ========================================================================
    v_lshlrev_b32_e32 v3, 12, v1              // wave_offset = wave_id * 4096
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
    s_mov_b32 s31, 0                          // k_tile_idx = 0
    
K_LOOP:
    // ========================================================================
    // COOPERATIVE K LOADING (all 4 waves contribute)
    // 8KB K data / 256 threads / 16 bytes = 2 loads per thread
    // ========================================================================
    s_lshl_b32 s33, s31, 13                   // k_offset = k_tile_idx * 8192
    v_mov_b32_e32 v10, s12                    // K base low
    v_mov_b32_e32 v11, s13                    // K base high
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10        // + thread offset (tid * 16)
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[80:83], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[84:87], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store K to LDS
    v_add_u32_e32 v7, LDS_K_OFFSET, v6
    ds_write_b128 v7, v[80:83]
    v_add_u32_e32 v7, LDS_K_OFFSET + 4096, v6
    ds_write_b128 v7, v[84:87]
    
    s_barrier
    
    // ========================================================================
    // INITIALIZE QK ACCUMULATOR (per-wave)
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
    
    // ========================================================================
    // Q×K GEMM (per-wave, each wave reads different Q rows from LDS)
    // ========================================================================
    // Q LDS offset: wave_offset + lane_id * 8
    v_lshlrev_b32_e32 v7, 3, v2               // lane_id * 8
    v_add_u32_e32 v7, v3, v7                  // + wave_offset
    
    // Load Q from LDS (K=0..31)
    ds_read_b64 a[0:1], v7                    // K=0..15
    v_add_u32_e32 v9, 16, v7
    ds_read_b64 a[2:3], v9                    // K=16..31
    
    // K LDS address (same for all waves - shared K data)
    v_lshlrev_b32_e32 v8, 3, v2               // lane_id * 8
    v_add_u32_e32 v9, LDS_K_OFFSET, v8
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs K=0..31
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=32..63 (wave-specific)
    v_add_u32_e32 v9, 32, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    
    // Load K K=32..63 (shared)
    v_add_u32_e32 v9, LDS_K_OFFSET + 32, v8
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    // Load Q K=64..95
    v_add_u32_e32 v9, 64, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[2:3], v9
    
    // Load K K=64..95
    v_add_u32_e32 v9, LDS_K_OFFSET + 64, v8
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=96..127
    v_add_u32_e32 v9, 96, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    
    // Load K K=96..127
    v_add_u32_e32 v9, LDS_K_OFFSET + 96, v8
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    s_nop 7
    
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
    // ONLINE SOFTMAX: Find local max
    // ========================================================================
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
    
    // Cross-lane max reduction
    v_mov_b32_e32 v19, v18
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v19, v18
    v_max_f32_e32 v18, v18, v19
    
    // Update running max
    v_mov_b32_e32 v19, v16
    v_max_f32_e32 v16, v16, v18
    
    // Correction factor
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
    
    // Sum P values
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
    
    // Cross-lane sum
    v_mov_b32_e32 v21, v20
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v21, v20
    v_add_f32_e32 v20, v20, v21
    
    v_add_f32_e32 v17, v17, v20
    
    s_nop 7
    
    // ========================================================================
    // PACK P TO FP8
    // ========================================================================
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
    
    // 1/sum for normalization
    v_rcp_f32_e32 v25, v17
    
    // ========================================================================
    // D-TILE LOOP
    // ========================================================================
    s_mov_b32 s40, 0
    
D_TILE_LOOP:
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
    
    // ========================================================================
    // COOPERATIVE V LOADING FOR THIS D-TILE
    // 4KB V data = 32 D × 128 K FP8 bytes
    // 256 threads × 16 bytes = 4KB (perfect fit!)
    // ========================================================================
    v_and_b32_e32 v80, 31, v0                 // d = tid % 32
    v_lshrrev_b32_e32 v81, 5, v0              // K_half = tid / 32 (0..7)
    
    // D_actual = D_tile * 32 + d
    v_mov_b32_e32 v82, s40
    v_lshlrev_b32_e32 v82, 5, v82
    v_add_u32_e32 v80, v82, v80
    
    // V address = V_base + k_offset + D_actual + K_section * 16 * 128
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33                     // k_offset
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v80, v10       // + D_actual
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // K_section = tid / 32 (0..7), each section handles 8 K rows
    // K_section * 8 * 128 = K_section * 1024
    v_lshlrev_b32_e32 v7, 10, v81
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 K values with stride 128
    flat_load_ubyte v72, v[10:11]
    flat_load_ubyte v73, v[10:11] offset:128
    flat_load_ubyte v74, v[10:11] offset:256
    flat_load_ubyte v75, v[10:11] offset:384
    flat_load_ubyte v76, v[10:11] offset:512
    flat_load_ubyte v77, v[10:11] offset:640
    flat_load_ubyte v78, v[10:11] offset:768
    flat_load_ubyte v79, v[10:11] offset:896
    
    s_waitcnt vmcnt(0)
    
    // Pack 8 bytes into 2 dwords
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
    
    // Store V to LDS with transposed layout: V[D, K] at offset d*64 + K_section*8
    v_and_b32_e32 v83, 31, v0                 // d = tid % 32
    v_lshlrev_b32_e32 v7, 6, v83              // d * 64 (stride for 64 K values in LDS)
    v_lshlrev_b32_e32 v8, 3, v81              // K_section * 8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_write_b64 v7, v[64:65]
    
    s_barrier
    
    // Write P FP8 to AGPRs
    v_accvgpr_write_b32 a0, v21
    v_accvgpr_write_b32 a1, v22
    v_accvgpr_write_b32 a2, v23
    v_accvgpr_write_b32 a3, v24
    
    // ========================================================================
    // P×V MFMA (all waves use same V data from LDS)
    // ========================================================================
    v_and_b32_e32 v7, 31, v2                  // D = lane % 32
    v_lshlrev_b32_e32 v7, 6, v7               // D * 64
    v_lshrrev_b32_e32 v8, 5, v2               // K_group = lane / 32
    v_lshlrev_b32_e32 v8, 3, v8               // K_group * 8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_read_b64 v[64:65], v7
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    // V for K=16..31
    v_and_b32_e32 v7, 31, v2
    v_lshlrev_b32_e32 v7, 6, v7
    v_lshrrev_b32_e32 v8, 5, v2
    v_lshlrev_b32_e32 v8, 3, v8
    v_add_u32_e32 v8, 16, v8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_read_b64 v[64:65], v7
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[64:65], v[48:63]
    
    // V for K=32..47
    v_and_b32_e32 v7, 31, v2
    v_lshlrev_b32_e32 v7, 6, v7
    v_lshrrev_b32_e32 v8, 5, v2
    v_lshlrev_b32_e32 v8, 3, v8
    v_add_u32_e32 v8, 32, v8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_read_b64 v[64:65], v7
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    // V for K=48..63
    v_and_b32_e32 v7, 31, v2
    v_lshlrev_b32_e32 v7, 6, v7
    v_lshrrev_b32_e32 v8, 5, v2
    v_lshlrev_b32_e32 v8, 3, v8
    v_add_u32_e32 v8, 48, v8
    v_add_u32_e32 v7, v8, v7
    v_add_u32_e32 v7, LDS_V_OFFSET, v7
    ds_read_b64 v[64:65], v7
    
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[64:65], v[48:63]
    
    s_nop 7
    
    // ========================================================================
    // NORMALIZE OUTPUT
    // ========================================================================
    v_mul_f32_e32 v48, v25, v48
    v_mul_f32_e32 v49, v25, v49
    v_mul_f32_e32 v50, v25, v50
    v_mul_f32_e32 v51, v25, v51
    v_mul_f32_e32 v52, v25, v52
    v_mul_f32_e32 v53, v25, v53
    v_mul_f32_e32 v54, v25, v54
    v_mul_f32_e32 v55, v25, v55
    v_mul_f32_e32 v56, v25, v56
    v_mul_f32_e32 v57, v25, v57
    v_mul_f32_e32 v58, v25, v58
    v_mul_f32_e32 v59, v25, v59
    v_mul_f32_e32 v60, v25, v60
    v_mul_f32_e32 v61, v25, v61
    v_mul_f32_e32 v62, v25, v62
    v_mul_f32_e32 v63, v25, v63
    
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
    // STORE OUTPUT (wave-specific offset)
    // Output layout matches v4 pattern but with wave offset
    // Base + wave_id * 16384 + D_tile * 4096 + lane_id * 64
    // (16384 = 64 lanes * 16 floats * 4 bytes * 4 D-tiles)
    // ========================================================================
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    // Wave offset: wave_id * 16384
    v_lshlrev_b32_e32 v7, 14, v1
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // D-tile offset: D_tile * 4096
    s_lshl_b32 s41, s40, 12
    v_mov_b32_e32 v7, s41
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Lane offset: lane_id * 64
    v_lshlrev_b32_e32 v7, 6, v2
    v_add_co_u32_e32 v10, vcc, v7, v10
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
    
    // D-tile loop
    s_add_u32 s40, s40, 1
    s_cmp_lt_u32 s40, 4
    s_cbranch_scc1 D_TILE_LOOP
    
    // K-loop
    s_add_u32 s31, s31, 1
    s_cmp_lt_u32 s31, s30
    s_cbranch_scc1 K_LOOP
    
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
    .amdhsa_next_free_vgpr 120
    .amdhsa_next_free_sgpr 50
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
  - .name: _ZN5aiter18fmha_fwd_hd128_fp8E
    .symbol: _ZN5aiter18fmha_fwd_hd128_fp8E.kd
    .kernarg_segment_size: 528
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 50
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

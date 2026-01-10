// SPDX-License-Identifier: MIT
// FP8 Flash Attention - V×P version (correct operand order)
// Based on BF16 kernel analysis: compute V×P instead of P×V
// A operand = V (AGPRs), B operand = P (VGPRs)
// Output = O^T[D, Q] transposed in store

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter22fmha_fwd_hd128_fp8_vxpE
.p2align 8
.type _ZN5aiter22fmha_fwd_hd128_fp8_vxpE,@function

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

_ZN5aiter22fmha_fwd_hd128_fp8_vxpE:
    // ========================================================================
    // REGISTER ALLOCATION:
    // s0-s1:   kernarg base pointer
    // s2:      workgroup_id_x (Q-tile index)
    // s3:      workgroup_id_y (head index) 
    // s4:      workgroup_id_z (batch index)
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
    // s29:     combined QK scale
    // s30:     k_tiles
    // s31:     current k_tile index
    // s32-s33: q_offset
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
    
    // ========================================================================
    // COMPUTE WORKGROUP OFFSETS
    // ========================================================================
    s_lshl_b32 s23, s21, 7                    // head_stride_q
    s_lshl_b32 s24, s22, 7                    // head_stride_kv
    s_lshl_b32 s32, s2, 13                    // q_offset = qtile_idx * 8192
    
    // ========================================================================
    // COMPUTE QK SCALE
    // ========================================================================
    v_mov_b32_e32 v3, s20
    v_mov_b32_e32 v4, s26
    v_mov_b32_e32 v5, s27
    v_mul_f32_e32 v3, v3, v4
    v_mul_f32_e32 v3, v3, v5
    v_readfirstlane_b32 s29, v3
    
    // Calculate k_tiles
    s_add_u32 s30, s22, 63
    s_lshr_b32 s30, s30, 6
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE
    // ========================================================================
    v_mov_b32_e32 v16, 0xff800000             // running_max = -inf
    v_mov_b32_e32 v17, 0                      // running_sum = 0
    
    // ========================================================================
    // INITIALIZE OUTPUT ACCUMULATORS
    // For V×P: output is O^T[D, Q], each thread owns 16 D values at one Q
    // ========================================================================
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

    // ========================================================================
    // LOAD Q TILE TO LDS
    // ========================================================================
    v_lshlrev_b32_e32 v6, 4, v0               // thread offset = tid * 16
    
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
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
    s_mov_b32 s31, 0
    
K_LOOP:
    // ========================================================================
    // LOAD K TILE TO LDS
    // ========================================================================
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
    
    // ========================================================================
    // INITIALIZE QK ACCUMULATOR
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
    // Q×K GEMM (same as before)
    // ========================================================================
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
    
    // Cross-thread max reduction
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
    // COMPUTE exp(QK - new_max) = P values
    // After this, v32-v47 contain P values (softmax output)
    // These will be used as B operand for V×P MFMA
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
    
    // Sum P values for running sum
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
    
    s_nop 7

    // ========================================================================
    // LOAD V AND PACK TO AGPRS (for V×P MFMA A operand)
    // V layout: V[K, D] in global memory, need V^T[D, K] for A operand
    //
    // MFMA A operand distribution for 32×32×16:
    // - Thread tid provides 8 K values at M=tid%32 position
    // - tid/32 determines which K range: 0=K[0..7], 1=K[8..15]
    //
    // For MFMA 1 (K=0..15):
    //   tid 0-31: A[M=0..31, K=0..7]
    //   tid 32-63: A[M=0..31, K=8..15]
    // For MFMA 2 (K=16..31):
    //   tid 0-31: A[M=0..31, K=16..23]
    //   tid 32-63: A[M=0..31, K=24..31]
    //
    // So each thread loads 8 K values at its D position:
    //   K_start = (tid/32) * 8 for MFMA1, K_start = 16 + (tid/32)*8 for MFMA2
    // ========================================================================
    
    v_and_b32_e32 v80, 31, v0             // d = tid % 32 (D position = M row)
    v_lshrrev_b32_e32 v81, 5, v0          // k_group = tid / 32 (0 or 1)
    
    // Global V base + k_tile_offset + D
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33                 // k_tile_offset
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v80, v10   // + D position
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // For MFMA 1: K_start = k_group * 8 (threads 0-31: K=0, threads 32-63: K=8)
    // K_start * 128 (stride) = k_group * 8 * 128 = k_group * 1024
    v_lshlrev_b32_e32 v7, 10, v81         // k_group * 1024
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 K values with stride 128 (head_dim) for MFMA 1 A operand
    flat_load_ubyte v72, v[10:11]                    // V[K=0 or 8, D=d]
    flat_load_ubyte v73, v[10:11] offset:128        // V[K=1 or 9, D=d]
    flat_load_ubyte v74, v[10:11] offset:256        // V[K=2 or 10, D=d]
    flat_load_ubyte v75, v[10:11] offset:384        // V[K=3 or 11, D=d]
    flat_load_ubyte v76, v[10:11] offset:512        // V[K=4 or 12, D=d]
    flat_load_ubyte v77, v[10:11] offset:640        // V[K=5 or 13, D=d]
    flat_load_ubyte v78, v[10:11] offset:768        // V[K=6 or 14, D=d]
    flat_load_ubyte v79, v[10:11] offset:896        // V[K=7 or 15, D=d]
    
    s_waitcnt vmcnt(0)
    
    // Pack 8 K values into 2 dwords for AGPR a[0:1]
    v_lshlrev_b32_e32 v73, 8, v73
    v_or_b32_e32 v72, v72, v73
    v_lshlrev_b32_e32 v74, 16, v74
    v_or_b32_e32 v72, v72, v74
    v_lshlrev_b32_e32 v75, 24, v75
    v_or_b32_e32 v64, v72, v75            // v64 = V[K=0..3, D=d] or V[K=8..11, D=d]
    
    v_lshlrev_b32_e32 v77, 8, v77
    v_or_b32_e32 v76, v76, v77
    v_lshlrev_b32_e32 v78, 16, v78
    v_or_b32_e32 v76, v76, v78
    v_lshlrev_b32_e32 v79, 24, v79
    v_or_b32_e32 v65, v76, v79            // v65 = V[K=4..7, D=d] or V[K=12..15, D=d]
    
    // Write V to AGPRs for MFMA 1
    v_accvgpr_write_b32 a0, v64
    v_accvgpr_write_b32 a1, v65
    
    // For MFMA 2: K_start = 16 + k_group * 8 (threads 0-31: K=16, threads 32-63: K=24)
    // Offset from current position: need to add (16 - k_group*8) * 128
    // For k_group=0: add 16*128 = 2048
    // For k_group=1: add (16-8)*128 = 8*128 = 1024
    // General: add (16 * 128) - (k_group * 8 * 128) = 2048 - k_group * 1024
    // But easier: just load from base + 16*128 + k_group*8*128
    
    // Recompute address for MFMA 2: V base + k_tile_offset + D + (16 + k_group*8)*128
    v_mov_b32_e32 v10, s14
    v_mov_b32_e32 v11, s15
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v80, v10   // + D position
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    // Add K=16 base offset: 16 * 128 = 2048
    v_add_co_u32_e32 v10, vcc, 2048, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    // Add k_group * 8 * 128 = k_group * 1024
    v_lshlrev_b32_e32 v7, 10, v81
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 K values for MFMA 2 A operand
    flat_load_ubyte v72, v[10:11]                    // V[K=16 or 24, D=d]
    flat_load_ubyte v73, v[10:11] offset:128
    flat_load_ubyte v74, v[10:11] offset:256
    flat_load_ubyte v75, v[10:11] offset:384
    flat_load_ubyte v76, v[10:11] offset:512
    flat_load_ubyte v77, v[10:11] offset:640
    flat_load_ubyte v78, v[10:11] offset:768
    flat_load_ubyte v79, v[10:11] offset:896
    
    s_waitcnt vmcnt(0)
    
    // Pack 8 K values into 2 dwords for AGPR a[2:3]
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

    // ========================================================================
    // PACK P VALUES TO FP8 IN VGPRS (B operand for V×P)
    // P values are in v32-v47 after softmax
    // Pack to FP8 and keep in VGPRs for B operand
    // ========================================================================
    
    // Pack P[0:3] into v82 (must use even register for MFMA B operand alignment)
    v_mov_b32_e32 v82, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v82, v32, v33
    v_cvt_pk_fp8_f32 v70, v34, v35
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v82, v82, v70
    
    // Pack P[4:7] into v83
    v_mov_b32_e32 v83, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v83, v36, v37
    v_cvt_pk_fp8_f32 v70, v38, v39
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v83, v83, v70
    
    // Pack P[8:11] into v84
    v_mov_b32_e32 v84, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v84, v40, v41
    v_cvt_pk_fp8_f32 v70, v42, v43
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v84, v84, v70
    
    // Pack P[12:15] into v85
    v_mov_b32_e32 v85, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v85, v44, v45
    v_cvt_pk_fp8_f32 v70, v46, v47
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v85, v85, v70
    
    s_nop 7

    // ========================================================================
    // V×P MFMA: A=V (AGPRs), B=P (VGPRs)
    // Computes O^T = V^T × P^T
    // Output is O^T[D, Q] - each thread owns 16 D values at one Q position
    // ========================================================================
    
    // MFMA 1: V[K=0..15] × P[K=0..15]
    // A operand: a[0:1] = V[D=tid%32, K=0..7 or K=8..15]
    // B operand: v[82:83] = P[Q_row_range, K=0..7 or K=8..15]
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[82:83], v[48:63]
    
    // MFMA 2: V[K=16..31] × P[K=16..31]
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[84:85], v[48:63]
    
    s_nop 7

    // ========================================================================
    // K-LOOP INCREMENT AND CHECK
    // ========================================================================
    s_add_u32 s31, s31, 1
    s_cmp_lt_u32 s31, s30
    s_cbranch_scc1 K_LOOP
    
    // ========================================================================
    // FINAL NORMALIZATION
    // ========================================================================
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
    // Output is O^T[D, Q] from V×P MFMA
    // Thread t owns O^T[D_base:D_base+16, Q=t%32]
    // Need to store as O[Q, D] = O[Q=t%32, D_base:D_base+16]
    //
    // Output address: output_ptr + Q_row * HEAD_DIM * sizeof(float) + D_base * sizeof(float)
    //               = output_ptr + (t%32) * 128 * 4 + (t/32) * 16 * 4
    //               = output_ptr + (t%32) * 512 + (t/32) * 64
    // ========================================================================
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    // Q_row = tid % 32
    v_and_b32_e32 v3, 31, v0
    // D_group = tid / 32 (0 or 1)
    v_lshrrev_b32_e32 v4, 5, v0
    
    // Q_row * 512
    v_lshlrev_b32_e32 v3, 9, v3
    // D_group * 64
    v_lshlrev_b32_e32 v4, 6, v4
    // Total offset
    v_add_u32_e32 v3, v3, v4
    
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store 16 floats (4 dwordx4 stores)
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

.size _ZN5aiter22fmha_fwd_hd128_fp8_vxpE, .-_ZN5aiter22fmha_fwd_hd128_fp8_vxpE

// ========================================================================
// KERNEL DESCRIPTOR
// ========================================================================
.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter22fmha_fwd_hd128_fp8_vxpE
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
  - .name: _ZN5aiter22fmha_fwd_hd128_fp8_vxpE
    .symbol: _ZN5aiter22fmha_fwd_hd128_fp8_vxpE.kd
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

// Full FP8 Flash Attention Kernel for HD=128
// O[32×128] = softmax(Q[32×128] @ K^T[128×32] / sqrt(d)) @ V[32×128]
//
// Design: FP8 for BMM only, F32 for everything else
// - QK MFMA: FP8 inputs → F32 accumulator
// - Softmax: F32 (with 1/sqrt(d) scaling)
// - PV MFMA: FP8 inputs → F32 accumulator
// - Output: F32
//
// This matches BF16 kernel behavior - only the MFMA data type changes.
// Inputs should be in natural range (e.g., [-3, 3] for normalized activations).
//
// Memory pattern: buffer_load → LDS → VGPR

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter14test_full_hd128E
.p2align 8
.type _ZN5aiter14test_full_hd128E,@function

_ZN5aiter14test_full_hd128E:
    s_mov_b64 exec, -1
    
    // Load kernel args
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [32×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [32×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [32×128] FP8
    
    v_and_b32_e32 v0, 63, v0
    
    // Softmax scaling: log2(e) / sqrt(128) = 1.4427 / 11.314 = 0.12754
    // This applies 1/sqrt(d) scaling implicitly in softmax
    s_mov_b32 s2, 0x3e028f5c              // log2(e) / sqrt(128)
    s_mov_b32 s3, 0xff800000              // -infinity
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // STAGE 1: QK MFMA - S[32×32] = Q[32×128] @ K^T[128×32]
    // ========================================================================
    
    // Load Q[32×128] to LDS at offset 0 (4KB)
    s_mov_b32 s10, 4096
    s_mov_b32 s11, 0x20000
    s_mov_b32 s20, 0
    v_lshlrev_b32_e32 v1, 4, v0
    
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 1024
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 2048
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 3072
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    // Load K[32×128] to LDS at offset 4096 (4KB)
    s_mov_b32 s14, 4096
    s_mov_b32 s15, 0x20000
    
    s_mov_b32 m0, 4096
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 5120
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 6144
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 7168
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
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
    
    // Thread indices for QK
    // COMPUTE S^T = K @ Q^T (not S = Q @ K^T)
    // This makes each thread hold one QUERY column of S^T
    // So summing VGPRs = summing across keys = row-wise softmax!
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4
    v_add_u32_e32 v6, 4096, v5            // K base at LDS offset 4096
    
    // 8 QK MFMA passes: S^T = K @ Q^T
    // A operand = K, B operand = Q
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \k_off, v6      // K offset (A operand)
        v_add_u32_e32 v8, \k_off, v5      // Q offset (B operand)
        ds_read_b64 v[20:21], v7          // Read K for A operand
        ds_read_b64 v[22:23], v8          // Read Q for B operand
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v20       // K → A
        v_accvgpr_write_b32 a1, v21
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
        s_nop 15
    .endr
    s_nop 15
    
    // ========================================================================
    // STAGE 2: TWO-PHASE SOFTMAX
    // ========================================================================
    
    // Initialize running state
    v_mov_b32_e32 v24, s3                 // running_max = -inf
    v_mov_b32_e32 v18, 0                  // running_sum
    
    // Phase 1a: Local max
    v_max_f32_e32 v21, v32, v33
    v_max_f32_e32 v21, v21, v34
    v_max_f32_e32 v21, v21, v35
    v_max_f32_e32 v21, v21, v36
    v_max_f32_e32 v21, v21, v37
    v_max_f32_e32 v21, v21, v38
    v_max_f32_e32 v21, v21, v39
    v_max_f32_e32 v21, v21, v40
    v_max_f32_e32 v21, v21, v41
    v_max_f32_e32 v21, v21, v42
    v_max_f32_e32 v21, v21, v43
    v_max_f32_e32 v21, v21, v44
    v_max_f32_e32 v21, v21, v45
    v_max_f32_e32 v21, v21, v46
    v_max_f32_e32 v21, v21, v47
    
    // Phase 1b: Cross-lane max (like BF16 - only permlane32_swap)
    // Each row's data is in one half-wavefront (32 threads)
    // permlane32_swap exchanges between halves for rows that span both
    v_mov_b32_e32 v20, v21
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21
    
    // Phase 1c: Correction factor
    v_sub_f32_e32 v16, v24, v21
    v_mul_f32_e32 v16, s2, v16
    v_exp_f32_e32 v16, v16
    s_nop 7
    v_max_f32_e32 v24, v24, v21
    v_mul_f32_e32 v23, s2, v21
    
    // Phase 1d: (S - max) * log2e
    v_fma_f32 v32, v32, s2, -v23
    v_fma_f32 v33, v33, s2, -v23
    v_fma_f32 v34, v34, s2, -v23
    v_fma_f32 v35, v35, s2, -v23
    v_fma_f32 v36, v36, s2, -v23
    v_fma_f32 v37, v37, s2, -v23
    v_fma_f32 v38, v38, s2, -v23
    v_fma_f32 v39, v39, s2, -v23
    v_fma_f32 v40, v40, s2, -v23
    v_fma_f32 v41, v41, s2, -v23
    v_fma_f32 v42, v42, s2, -v23
    v_fma_f32 v43, v43, s2, -v23
    v_fma_f32 v44, v44, s2, -v23
    v_fma_f32 v45, v45, s2, -v23
    v_fma_f32 v46, v46, s2, -v23
    v_fma_f32 v47, v47, s2, -v23
    
    // Phase 1e: exp
    v_exp_f32_e32 v32, v32
    v_exp_f32_e32 v33, v33
    v_exp_f32_e32 v34, v34
    v_exp_f32_e32 v35, v35
    v_exp_f32_e32 v36, v36
    v_exp_f32_e32 v37, v37
    v_exp_f32_e32 v38, v38
    v_exp_f32_e32 v39, v39
    v_exp_f32_e32 v40, v40
    v_exp_f32_e32 v41, v41
    v_exp_f32_e32 v42, v42
    v_exp_f32_e32 v43, v43
    v_exp_f32_e32 v44, v44
    v_exp_f32_e32 v45, v45
    v_exp_f32_e32 v46, v46
    v_exp_f32_e32 v47, v47
    s_nop 7
    
    // Phase 2: Sum for row-wise softmax
    // With S^T = K @ Q^T layout:
    // - Thread t holds all keys' P values for query t%32
    // - v32-v47 hold 16 different key rows for this query
    // - Sum VGPRs = partial sum of 16 keys
    // - Threads 0 and 32 both have query 0, just different key subsets
    // - permlane32_swap combines the two halves for each query
    // - NO ds_swizzle! That would mix different queries!
    
    v_add_f32_e32 v20, v32, v33
    v_add_f32_e32 v20, v20, v34
    v_add_f32_e32 v20, v20, v35
    v_add_f32_e32 v20, v20, v36
    v_add_f32_e32 v20, v20, v37
    v_add_f32_e32 v20, v20, v38
    v_add_f32_e32 v20, v20, v39
    v_add_f32_e32 v20, v20, v40
    v_add_f32_e32 v20, v20, v41
    v_add_f32_e32 v20, v20, v42
    v_add_f32_e32 v20, v20, v43
    v_add_f32_e32 v20, v20, v44
    v_add_f32_e32 v20, v20, v45
    v_add_f32_e32 v20, v20, v46
    v_add_f32_e32 v20, v20, v47
    // v20 = partial sum of 16 keys for this query
    
    // Combine with paired thread (0↔32, 1↔33, etc.)
    // Both threads hold the SAME query, different key subsets
    v_mov_b32_e32 v22, v20
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v22, v20
    v_add_f32_e32 v18, v20, v22
    // v18 = sum of all 32 keys for this query (row-wise sum!)
    
    // Normalize P NOW (before FP8 conversion)
    // This ensures sum(P) = 1.0 exactly, avoiding FP8 quantization mismatch
    v_rcp_f32_e32 v19, v18
    s_nop 3
    v_mul_f32_e32 v32, v32, v19
    v_mul_f32_e32 v33, v33, v19
    v_mul_f32_e32 v34, v34, v19
    v_mul_f32_e32 v35, v35, v19
    v_mul_f32_e32 v36, v36, v19
    v_mul_f32_e32 v37, v37, v19
    v_mul_f32_e32 v38, v38, v19
    v_mul_f32_e32 v39, v39, v19
    v_mul_f32_e32 v40, v40, v19
    v_mul_f32_e32 v41, v41, v19
    v_mul_f32_e32 v42, v42, v19
    v_mul_f32_e32 v43, v43, v19
    v_mul_f32_e32 v44, v44, v19
    v_mul_f32_e32 v45, v45, v19
    v_mul_f32_e32 v46, v46, v19
    v_mul_f32_e32 v47, v47, v19
    // Now P (in v32-v47) is normalized: sum = 1.0 for each query
    
    // ========================================================================
    // STAGE 3: Store P to LDS (TRANSPOSED), Load V
    // ========================================================================
    s_barrier
    
    // We have P^T[key, query] in VGPRs from S^T computation
    // Thread t holds P^T[key_interleaved, query=t%32]
    // Need to store as P[query, key] for PV MFMA
    // So: row = query = tid%32, col = key_interleaved
    
    v_and_b32_e32 v2, 31, v0              // query = tid % 32
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v3, 2, v3           // (tid/32) * 4 for key offset
    
    .macro STORE_P vreg, key_mod4, key_8_group
        // Compute key index (was row, now becomes column)
        v_mov_b32_e32 v7, \key_mod4
        v_add_u32_e32 v7, v7, v3          // + (tid/32)*4
        v_add_u32_e32 v7, \key_8_group * 8, v7  // + key_8_group * 8
        // Store at P[query, key] = P[row=v2, col=v7]
        // Address = row * 32 + col = query * 32 + key
        v_lshlrev_b32_e32 v8, 5, v2       // query * 32
        v_add_u32_e32 v8, v8, v7          // + key
        v_lshlrev_b32_e32 v8, 2, v8       // byte offset
        ds_write_b32 v8, \vreg
    .endm
    
    STORE_P v32, 0, 0
    STORE_P v33, 1, 0
    STORE_P v34, 2, 0
    STORE_P v35, 3, 0
    STORE_P v36, 0, 1
    STORE_P v37, 1, 1
    STORE_P v38, 2, 1
    STORE_P v39, 3, 1
    STORE_P v40, 0, 2
    STORE_P v41, 1, 2
    STORE_P v42, 2, 2
    STORE_P v43, 3, 2
    STORE_P v44, 0, 3
    STORE_P v45, 1, 3
    STORE_P v46, 2, 3
    STORE_P v47, 3, 3
    
    // Load V[32×128] to LDS at offset 4096 (4KB)
    s_mov_b32 s18, 4096
    s_mov_b32 s19, 0x20000
    v_lshlrev_b32_e32 v1, 4, v0
    
    s_mov_b32 m0, 4096
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    s_mov_b32 m0, 5120
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    s_mov_b32 m0, 6144
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    s_mov_b32 m0, 7168
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    
    s_waitcnt lgkmcnt(0) vmcnt(0)
    s_barrier
    
    // ========================================================================
    // STAGE 4: PV MFMA - O[32×128] = P[32×32] @ V[32×128]
    // 4 output tiles × 2 K-passes = 8 MFMAs
    // ========================================================================
    
    // Macro for one output tile
    .macro PV_OUTPUT_TILE tile_col_offset
        v_and_b32_e32 v2, 31, v0
        v_lshrrev_b32_e32 v3, 5, v0
        
        // Init O accumulator
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
        
        // K-pass 0: k=0..15
        v_lshlrev_b32_e32 v4, 7, v2           // row * 128 (P is 32×32×4)
        v_lshlrev_b32_e32 v5, 5, v3           // half * 32
        v_add_u32_e32 v4, v4, v5
        
        ds_read_b64 v[20:21], v4
        v_add_u32_e32 v5, 8, v4
        ds_read_b64 v[22:23], v5
        v_add_u32_e32 v5, 16, v4
        ds_read_b64 v[24:25], v5
        v_add_u32_e32 v5, 24, v4
        ds_read_b64 v[26:27], v5
        s_waitcnt lgkmcnt(0)
        
        v_cvt_pk_fp8_f32 v28, v20, v21
        v_and_b32_e32 v28, 0xFFFF, v28
        v_cvt_pk_fp8_f32 v29, v22, v23
        v_lshlrev_b32_e32 v29, 16, v29
        v_and_b32_e32 v29, 0xFFFF0000, v29
        v_or_b32_e32 v28, v28, v29
        v_cvt_pk_fp8_f32 v29, v24, v25
        v_and_b32_e32 v29, 0xFFFF, v29
        v_cvt_pk_fp8_f32 v30, v26, v27
        v_lshlrev_b32_e32 v30, 16, v30
        v_and_b32_e32 v30, 0xFFFF0000, v30
        v_or_b32_e32 v29, v29, v30
        
        v_lshlrev_b32_e32 v5, 10, v3
        v_add_u32_e32 v5, v5, v2
        v_add_u32_e32 v5, 4096 + \tile_col_offset, v5
        
        ds_read_u8 v10, v5
        v_add_u32_e32 v6, 128, v5
        ds_read_u8 v11, v6
        v_add_u32_e32 v6, 256, v5
        ds_read_u8 v12, v6
        v_add_u32_e32 v6, 384, v5
        ds_read_u8 v13, v6
        v_add_u32_e32 v6, 512, v5
        ds_read_u8 v14, v6
        v_add_u32_e32 v6, 640, v5
        ds_read_u8 v15, v6
        v_add_u32_e32 v6, 768, v5
        ds_read_u8 v16, v6
        v_add_u32_e32 v6, 896, v5
        ds_read_u8 v17, v6
        s_waitcnt lgkmcnt(0)
        
        v_lshlrev_b32_e32 v11, 8, v11
        v_or_b32_e32 v10, v10, v11
        v_lshlrev_b32_e32 v12, 16, v12
        v_or_b32_e32 v10, v10, v12
        v_lshlrev_b32_e32 v13, 24, v13
        v_or_b32_e32 v30, v10, v13
        v_lshlrev_b32_e32 v15, 8, v15
        v_or_b32_e32 v14, v14, v15
        v_lshlrev_b32_e32 v16, 16, v16
        v_or_b32_e32 v14, v14, v16
        v_lshlrev_b32_e32 v17, 24, v17
        v_or_b32_e32 v31, v14, v17
        
        v_accvgpr_write_b32 a0, v28
        v_accvgpr_write_b32 a1, v29
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[30:31], v[48:63]
        s_nop 15
        
        // K-pass 1: k=16..31
        v_add_u32_e32 v4, 64, v4
        ds_read_b64 v[20:21], v4
        v_add_u32_e32 v5, 8, v4
        ds_read_b64 v[22:23], v5
        v_add_u32_e32 v5, 16, v4
        ds_read_b64 v[24:25], v5
        v_add_u32_e32 v5, 24, v4
        ds_read_b64 v[26:27], v5
        s_waitcnt lgkmcnt(0)
        
        v_cvt_pk_fp8_f32 v28, v20, v21
        v_and_b32_e32 v28, 0xFFFF, v28
        v_cvt_pk_fp8_f32 v29, v22, v23
        v_lshlrev_b32_e32 v29, 16, v29
        v_and_b32_e32 v29, 0xFFFF0000, v29
        v_or_b32_e32 v28, v28, v29
        v_cvt_pk_fp8_f32 v29, v24, v25
        v_and_b32_e32 v29, 0xFFFF, v29
        v_cvt_pk_fp8_f32 v30, v26, v27
        v_lshlrev_b32_e32 v30, 16, v30
        v_and_b32_e32 v30, 0xFFFF0000, v30
        v_or_b32_e32 v29, v29, v30
        
        v_lshlrev_b32_e32 v5, 10, v3
        v_add_u32_e32 v5, v5, v2
        v_add_u32_e32 v5, 4096 + 2048 + \tile_col_offset, v5
        
        ds_read_u8 v10, v5
        v_add_u32_e32 v6, 128, v5
        ds_read_u8 v11, v6
        v_add_u32_e32 v6, 256, v5
        ds_read_u8 v12, v6
        v_add_u32_e32 v6, 384, v5
        ds_read_u8 v13, v6
        v_add_u32_e32 v6, 512, v5
        ds_read_u8 v14, v6
        v_add_u32_e32 v6, 640, v5
        ds_read_u8 v15, v6
        v_add_u32_e32 v6, 768, v5
        ds_read_u8 v16, v6
        v_add_u32_e32 v6, 896, v5
        ds_read_u8 v17, v6
        s_waitcnt lgkmcnt(0)
        
        v_lshlrev_b32_e32 v11, 8, v11
        v_or_b32_e32 v10, v10, v11
        v_lshlrev_b32_e32 v12, 16, v12
        v_or_b32_e32 v10, v10, v12
        v_lshlrev_b32_e32 v13, 24, v13
        v_or_b32_e32 v30, v10, v13
        v_lshlrev_b32_e32 v15, 8, v15
        v_or_b32_e32 v14, v14, v15
        v_lshlrev_b32_e32 v16, 16, v16
        v_or_b32_e32 v14, v14, v16
        v_lshlrev_b32_e32 v17, 24, v17
        v_or_b32_e32 v31, v14, v17
        
        v_accvgpr_write_b32 a0, v28
        v_accvgpr_write_b32 a1, v29
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[30:31], v[48:63]
        s_nop 15
        s_nop 15
    .endm
    
    // Store output tile macro
    .macro STORE_O tile_col_offset, vreg, row_mod4, row_8_group
        v_mov_b32_e32 v7, \row_mod4
        v_add_u32_e32 v7, v7, v3
        v_add_u32_e32 v7, \row_8_group * 8, v7
        v_lshlrev_b32_e32 v7, 7, v7
        v_add_u32_e32 v7, v7, v2
        v_add_u32_e32 v7, \tile_col_offset, v7
        v_lshlrev_b32_e32 v7, 2, v7
        v_mov_b32_e32 v10, s4
        v_mov_b32_e32 v11, s5
        v_add_co_u32_e32 v10, vcc, v7, v10
        v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
        flat_store_dword v[10:11], \vreg
    .endm
    
    // P is already normalized (sum = 1.0 per query)
    // No need for output normalization
    
    // Process all 4 output tiles
    .irp tile_off, 0, 32, 64, 96
        PV_OUTPUT_TILE \tile_off
        
        v_lshrrev_b32_e32 v3, 5, v0
        v_lshlrev_b32_e32 v3, 2, v3
        
        STORE_O \tile_off, v48, 0, 0
        STORE_O \tile_off, v49, 1, 0
        STORE_O \tile_off, v50, 2, 0
        STORE_O \tile_off, v51, 3, 0
        STORE_O \tile_off, v52, 0, 1
        STORE_O \tile_off, v53, 1, 1
        STORE_O \tile_off, v54, 2, 1
        STORE_O \tile_off, v55, 3, 1
        STORE_O \tile_off, v56, 0, 2
        STORE_O \tile_off, v57, 1, 2
        STORE_O \tile_off, v58, 2, 2
        STORE_O \tile_off, v59, 3, 2
        STORE_O \tile_off, v60, 0, 3
        STORE_O \tile_off, v61, 1, 3
        STORE_O \tile_off, v62, 2, 3
        STORE_O \tile_off, v63, 3, 3
    .endr
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter14test_full_hd128E
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 68
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 64
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter14test_full_hd128E
    .symbol: _ZN5aiter14test_full_hd128E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 68
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

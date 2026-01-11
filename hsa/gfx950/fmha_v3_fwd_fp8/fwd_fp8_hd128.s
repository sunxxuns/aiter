// FP8 Flash Attention HD=128 Kernel
// O[32×128] = softmax(Q[32×128] @ K^T[128×32]) @ V[32×128]
// Uses buffer_load→LDS pattern proven in test_qk_fixed.s
// Standard softmax (works for single K-tile, seq_len=32)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter14fwd_fp8_hd128E
.p2align 8
.type _ZN5aiter14fwd_fp8_hd128E,@function

_ZN5aiter14fwd_fp8_hd128E:
    s_mov_b64 exec, -1
    
    // Load kernel args: O, Q, K, V
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [32×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [32×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [32×128] FP8
    
    v_and_b32_e32 v0, 63, v0   // tid = threadIdx.x & 63
    
    // Constants
    s_mov_b32 s2, 0x3fb8aa3b   // log2(e)
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // STEP 1: Load Q[32×128] to LDS (4KB at offset 0)
    // 64 threads × 64 bytes each = 4096 bytes
    // Each thread does 4× buffer_load_dwordx4 (4 × 16 = 64 bytes)
    // ========================================================================
    s_mov_b32 s10, 4096        // num_records = 4KB
    s_mov_b32 s11, 0x20000     // buffer flags
    
    v_lshlrev_b32_e32 v1, 6, v0  // offset = tid * 64
    s_mov_b32 m0, 0              // LDS base = 0
    buffer_load_dwordx4 v1, s[8:11], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024       // LDS offset += 1024
    buffer_load_dwordx4 v1, s[8:11], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[8:11], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[8:11], 0 offen lds
    
    // ========================================================================
    // STEP 2: Load K[32×128] to LDS (4KB at offset 4096)
    // ========================================================================
    s_mov_b32 s14, 4096
    s_mov_b32 s15, 0x20000
    
    v_lshlrev_b32_e32 v1, 6, v0  // offset = tid * 64
    s_mov_b32 m0, 4096           // LDS base = 4096
    buffer_load_dwordx4 v1, s[12:15], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[12:15], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[12:15], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[12:15], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // STEP 3: QK MFMA - 8 passes, accumulate S[32×32] in v[32:47]
    // S = Q @ K^T where Q[32×128], K[32×128], S[32×32]
    // Each pass covers 16 elements of K-dimension
    // ========================================================================
    
    // Initialize S accumulators to 0
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
    
    // Calculate LDS read offsets
    // Q layout in LDS: 32 rows × 128 cols, row-major, FP8
    // For MFMA: need Q[row, k:k+16] and K[col, k:k+16]
    // Thread mapping: tid%32 = col, tid/32 = row_group (0 or 1)
    v_and_b32_e32 v2, 31, v0              // col = tid % 32
    v_lshrrev_b32_e32 v3, 5, v0           // row_group = tid / 32
    
    // Q LDS offset: row * 128 + k_offset
    // For thread groups: rows 0-3 + row_group*4 and rows 8-11 + row_group*4, etc
    // Actually for ds_read_b64 we read 8 bytes = 8 FP8 = half of 16-element K-tile
    v_lshlrev_b32_e32 v4, 3, v3           // row_group * 8
    v_lshlrev_b32_e32 v5, 7, v2           // col * 128 (K dimension stride)
    v_add_u32_e32 v5, v5, v4              // base Q offset (will add k_offset)
    
    // K LDS offset: starts at 4096
    v_add_u32_e32 v6, 4096, v5            // base K offset
    
    // MFMA pass 0: k=0..15
    ds_read_b64 v[20:21], v5
    ds_read_b64 v[22:23], v6
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 1: k=16..31
    v_add_u32_e32 v7, 16, v5
    v_add_u32_e32 v8, 16, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 2: k=32..47
    v_add_u32_e32 v7, 32, v5
    v_add_u32_e32 v8, 32, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 3: k=48..63
    v_add_u32_e32 v7, 48, v5
    v_add_u32_e32 v8, 48, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 4: k=64..79
    v_add_u32_e32 v7, 64, v5
    v_add_u32_e32 v8, 64, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 5: k=80..95
    v_add_u32_e32 v7, 80, v5
    v_add_u32_e32 v8, 80, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 6: k=96..111
    v_add_u32_e32 v7, 96, v5
    v_add_u32_e32 v8, 96, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 7: k=112..127
    v_add_u32_e32 v7, 112, v5
    v_add_u32_e32 v8, 112, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    s_nop 15
    
    // ========================================================================
    // STEP 4: Standard Softmax - P = softmax(S) row-wise
    // Each thread owns elements in specific rows based on MFMA output layout
    // ========================================================================
    
    // Step 4.1: Find row max across all 32 columns
    // Each thread has elements from its column (tid%32)
    // Need to reduce across threads 0-31 and 32-63 separately
    
    // Local max of my 16 elements (4 per row_group × 4 row_groups)
    v_max_f32_e32 v48, v32, v33
    v_max_f32_e32 v48, v48, v34
    v_max_f32_e32 v48, v48, v35
    v_max_f32_e32 v49, v36, v37
    v_max_f32_e32 v49, v49, v38
    v_max_f32_e32 v49, v49, v39
    v_max_f32_e32 v50, v40, v41
    v_max_f32_e32 v50, v50, v42
    v_max_f32_e32 v50, v50, v43
    v_max_f32_e32 v51, v44, v45
    v_max_f32_e32 v51, v51, v46
    v_max_f32_e32 v51, v51, v47
    
    // Max across all 16 elements per thread
    v_max_f32_e32 v48, v48, v49
    v_max_f32_e32 v48, v48, v50
    v_max_f32_e32 v48, v48, v51
    
    // Cross-thread max reduction using ds_swizzle
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x801F  // SWAP 16
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x401F  // SWAP 8
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x201F  // SWAP 4
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x101F  // SWAP 2
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x081F  // SWAP 1
    s_waitcnt lgkmcnt(0)
    v_max_f32_e32 v48, v48, v49
    
    // v48 now has max across all 32 columns (for threads 0-31 and 32-63 independently)
    // Broadcast to other half of wavefront
    v_readfirstlane_b32 s20, v48
    v_mov_b32_e32 v49, s20
    
    // Step 4.2: Compute exp(S - max) * log2(e)
    v_sub_f32_e32 v32, v32, v48
    v_sub_f32_e32 v33, v33, v48
    v_sub_f32_e32 v34, v34, v48
    v_sub_f32_e32 v35, v35, v48
    v_sub_f32_e32 v36, v36, v48
    v_sub_f32_e32 v37, v37, v48
    v_sub_f32_e32 v38, v38, v48
    v_sub_f32_e32 v39, v39, v48
    v_sub_f32_e32 v40, v40, v48
    v_sub_f32_e32 v41, v41, v48
    v_sub_f32_e32 v42, v42, v48
    v_sub_f32_e32 v43, v43, v48
    v_sub_f32_e32 v44, v44, v48
    v_sub_f32_e32 v45, v45, v48
    v_sub_f32_e32 v46, v46, v48
    v_sub_f32_e32 v47, v47, v48
    
    // Multiply by log2(e) and exp
    v_mul_f32_e32 v32, s2, v32
    v_mul_f32_e32 v33, s2, v33
    v_mul_f32_e32 v34, s2, v34
    v_mul_f32_e32 v35, s2, v35
    v_mul_f32_e32 v36, s2, v36
    v_mul_f32_e32 v37, s2, v37
    v_mul_f32_e32 v38, s2, v38
    v_mul_f32_e32 v39, s2, v39
    v_mul_f32_e32 v40, s2, v40
    v_mul_f32_e32 v41, s2, v41
    v_mul_f32_e32 v42, s2, v42
    v_mul_f32_e32 v43, s2, v43
    v_mul_f32_e32 v44, s2, v44
    v_mul_f32_e32 v45, s2, v45
    v_mul_f32_e32 v46, s2, v46
    v_mul_f32_e32 v47, s2, v47
    
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
    
    // Step 4.3: Compute row sum
    v_add_f32_e32 v48, v32, v33
    v_add_f32_e32 v48, v48, v34
    v_add_f32_e32 v48, v48, v35
    v_add_f32_e32 v49, v36, v37
    v_add_f32_e32 v49, v49, v38
    v_add_f32_e32 v49, v49, v39
    v_add_f32_e32 v50, v40, v41
    v_add_f32_e32 v50, v50, v42
    v_add_f32_e32 v50, v50, v43
    v_add_f32_e32 v51, v44, v45
    v_add_f32_e32 v51, v51, v46
    v_add_f32_e32 v51, v51, v47
    v_add_f32_e32 v48, v48, v49
    v_add_f32_e32 v48, v48, v50
    v_add_f32_e32 v48, v48, v51
    
    // Cross-thread sum reduction
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x801F
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x401F
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x201F
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x101F
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v48, v48, v49
    
    v_mov_b32_e32 v49, v48
    ds_swizzle_b32 v49, v49 offset:0x081F
    s_waitcnt lgkmcnt(0)
    v_add_f32_e32 v48, v48, v49
    
    // Broadcast sum
    v_readfirstlane_b32 s20, v48
    v_mov_b32_e32 v48, s20
    
    // Step 4.4: Divide by sum (P = exp(S-max) / sum)
    v_rcp_f32_e32 v48, v48
    s_nop 7
    
    v_mul_f32_e32 v32, v32, v48
    v_mul_f32_e32 v33, v33, v48
    v_mul_f32_e32 v34, v34, v48
    v_mul_f32_e32 v35, v35, v48
    v_mul_f32_e32 v36, v36, v48
    v_mul_f32_e32 v37, v37, v48
    v_mul_f32_e32 v38, v38, v48
    v_mul_f32_e32 v39, v39, v48
    v_mul_f32_e32 v40, v40, v48
    v_mul_f32_e32 v41, v41, v48
    v_mul_f32_e32 v42, v42, v48
    v_mul_f32_e32 v43, v43, v48
    v_mul_f32_e32 v44, v44, v48
    v_mul_f32_e32 v45, v45, v48
    v_mul_f32_e32 v46, v46, v48
    v_mul_f32_e32 v47, v47, v48
    
    // P is now in v[32:47] - same layout as S
    
    // ========================================================================
    // STEP 5: P Redistribution via LDS
    // Store P to LDS, then read back in PV layout
    // P[32×32] needs 32×32×4 = 4KB
    // ========================================================================
    s_barrier
    
    // Store P to LDS using correct MFMA output layout
    v_and_b32_e32 v2, 31, v0              // col = tid % 32
    v_lshrrev_b32_e32 v3, 5, v0           // tid // 32
    v_lshlrev_b32_e32 v3, 2, v3           // (tid // 32) * 4
    
    .macro STORE_P_LDS vreg, row_mod4, row_8_group
        v_mov_b32_e32 v7, \row_mod4
        v_add_u32_e32 v7, v7, v3
        v_add_u32_e32 v7, \row_8_group * 8, v7
        v_lshlrev_b32_e32 v7, 5, v7        // row * 32
        v_add_u32_e32 v7, v7, v2           // row * 32 + col
        v_lshlrev_b32_e32 v7, 2, v7        // byte offset
        ds_write_b32 v7, \vreg
    .endm
    
    STORE_P_LDS v32, 0, 0
    STORE_P_LDS v33, 1, 0
    STORE_P_LDS v34, 2, 0
    STORE_P_LDS v35, 3, 0
    STORE_P_LDS v36, 0, 1
    STORE_P_LDS v37, 1, 1
    STORE_P_LDS v38, 2, 1
    STORE_P_LDS v39, 3, 1
    STORE_P_LDS v40, 0, 2
    STORE_P_LDS v41, 1, 2
    STORE_P_LDS v42, 2, 2
    STORE_P_LDS v43, 3, 2
    STORE_P_LDS v44, 0, 3
    STORE_P_LDS v45, 1, 3
    STORE_P_LDS v46, 2, 3
    STORE_P_LDS v47, 3, 3
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // STEP 6: Load V[32×128] to LDS (reuse Q/K space at offset 4096)
    // ========================================================================
    s_mov_b32 s18, 4096
    s_mov_b32 s19, 0x20000
    
    v_lshlrev_b32_e32 v1, 6, v0  // offset = tid * 64
    s_mov_b32 m0, 4096           // LDS base = 4096 (after P)
    buffer_load_dwordx4 v1, s[16:19], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[16:19], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[16:19], 0 offen lds
    v_add_u32_e32 v1, 16, v1
    s_add_u32 m0, m0, 1024
    buffer_load_dwordx4 v1, s[16:19], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // STEP 7: PV MFMA - For each output tile, compute O_tile = P @ V_tile
    // O[32×128] = P[32×32] @ V[32×128]
    // We compute and store one 32×16 output tile at a time (8 tiles total)
    // ========================================================================
    
    // Read P from LDS for PV MFMA (need to convert F32 → FP8)
    // P layout in LDS: row-major 32×32×4 bytes
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v4, 3, v3           // row_group * 8
    v_lshlrev_b32_e32 v5, 5, v2           // col * 32 (stride in P)
    v_add_u32_e32 v5, v5, v4
    v_lshlrev_b32_e32 v5, 2, v5           // byte offset for F32
    
    // Read P row
    ds_read_b64 v[20:21], v5              // 2 F32 values
    v_add_u32_e32 v6, 8, v5
    ds_read_b64 v[22:23], v6              // next 2 F32 values
    s_waitcnt lgkmcnt(0)
    
    // Convert P F32 → FP8 (pack 4 F32 → 4 FP8 in one DWORD)
    // v_cvt_pk_fp8_f32 produces 2 FP8 in low 16 bits
    v_cvt_pk_fp8_f32 v24, v20, v21
    v_and_b32_e32 v24, 0xFFFF, v24        // mask garbage
    v_cvt_pk_fp8_f32 v25, v22, v23
    v_and_b32_e32 v25, 0xFFFF, v25
    v_lshlrev_b32_e32 v25, 16, v25
    v_or_b32_e32 v24, v24, v25            // 4 FP8 values in v24
    
    // Continue reading P and converting...
    v_add_u32_e32 v6, 16, v5
    ds_read_b64 v[20:21], v6
    v_add_u32_e32 v6, 24, v5
    ds_read_b64 v[22:23], v6
    s_waitcnt lgkmcnt(0)
    v_cvt_pk_fp8_f32 v25, v20, v21
    v_and_b32_e32 v25, 0xFFFF, v25
    v_cvt_pk_fp8_f32 v26, v22, v23
    v_and_b32_e32 v26, 0xFFFF, v26
    v_lshlrev_b32_e32 v26, 16, v26
    v_or_b32_e32 v25, v25, v26            // next 4 FP8 values in v25
    
    // P is packed: v24 has first 4 FP8, v25 has next 4 FP8
    // For MFMA we need 8 FP8 = 64 bits = 2 DWORDs
    // v[24:25] now has 8 consecutive P values for one row segment
    
    // For simplicity, we'll do a single output tile (first 32×16 of O)
    // Full implementation would loop over all 8 tiles
    
    // Read V tile from LDS
    // V layout: 32 rows × 128 cols, FP8, at LDS offset 4096
    v_lshlrev_b32_e32 v6, 7, v2           // col * 128 (row stride)
    v_add_u32_e32 v6, v6, v4              // + row_group * 8
    v_add_u32_e32 v6, 4096, v6            // + V base offset
    
    ds_read_b64 v[26:27], v6              // V tile data
    s_waitcnt lgkmcnt(0)
    
    // Initialize O accumulator for this tile
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
    
    // PV MFMA pass 1: k=0..15 (first 16 columns of P, first 16 rows of V)
    v_accvgpr_write_b32 a0, v24
    v_accvgpr_write_b32 a1, v25
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[26:27], v[48:63]
    s_nop 15
    
    // PV MFMA pass 2: k=16..31 (next 16 columns of P, next 16 rows of V)
    // For full correctness, need to read more P and V data...
    // This is simplified - full kernel needs all 2 passes for 32 P columns
    
    s_nop 15
    
    // ========================================================================
    // STEP 8: Store Output (first tile only for now)
    // ========================================================================
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v3, 2, v3
    
    .macro STORE_O vreg, row_mod4, row_8_group
        v_mov_b32_e32 v7, \row_mod4
        v_add_u32_e32 v7, v7, v3
        v_add_u32_e32 v7, \row_8_group * 8, v7
        v_lshlrev_b32_e32 v7, 7, v7        // row * 128 (output stride for HD=128)
        v_add_u32_e32 v7, v7, v2           // + col
        v_lshlrev_b32_e32 v7, 2, v7        // byte offset
        v_mov_b32_e32 v10, s4
        v_mov_b32_e32 v11, s5
        v_add_co_u32_e32 v10, vcc, v7, v10
        v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
        flat_store_dword v[10:11], \vreg
    .endm
    
    STORE_O v48, 0, 0
    STORE_O v49, 1, 0
    STORE_O v50, 2, 0
    STORE_O v51, 3, 0
    STORE_O v52, 0, 1
    STORE_O v53, 1, 1
    STORE_O v54, 2, 1
    STORE_O v55, 3, 1
    STORE_O v56, 0, 2
    STORE_O v57, 1, 2
    STORE_O v58, 2, 2
    STORE_O v59, 3, 2
    STORE_O v60, 0, 3
    STORE_O v61, 1, 3
    STORE_O v62, 2, 3
    STORE_O v63, 3, 3
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter14fwd_fp8_hd128E
    .amdhsa_group_segment_fixed_size 12288
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
  - .name: _ZN5aiter14fwd_fp8_hd128E
    .symbol: _ZN5aiter14fwd_fp8_hd128E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 12288
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

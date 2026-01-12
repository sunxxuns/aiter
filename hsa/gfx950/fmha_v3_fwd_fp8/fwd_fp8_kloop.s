// FP8 Flash Attention with K-tile Loop and Online Softmax
// O[32×128] = softmax(Q @ K^T / sqrt(d)) @ V
//
// Online softmax algorithm:
//   running_max = -inf, running_sum = 0, O = 0
//   for each K-tile:
//     S = K_tile @ Q^T (transpose trick)
//     tile_max = max(S)
//     correction = exp((running_max - new_max) * scale)
//     O = O * correction
//     running_sum = running_sum * correction
//     P = exp((S - tile_max) * scale)  // unnormalized
//     tile_sum = sum(P)
//     running_sum += tile_sum
//     running_max = new_max
//     O += P @ V_tile
//   O = O / running_sum
//
// Args: O[32×128], Q[32×128], K[seq×128], V[seq×128], seq_len

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13fwd_fp8_kloopE
.p2align 8
.type _ZN5aiter13fwd_fp8_kloopE,@function

_ZN5aiter13fwd_fp8_kloopE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [32×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [seq×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [seq×128] FP8
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    v_and_b32_e32 v0, 63, v0
    
    // Constants
    s_mov_b32 s2, 0x3e028f5c              // log2(e) / sqrt(128) = 0.12754
    s_mov_b32 s3, 0xff800000              // -infinity
    
    s_waitcnt lgkmcnt(0)
    
    // Calculate number of K-tiles
    s_add_i32 s25, s24, 31
    s_lshr_b32 s25, s25, 5                 // num_tiles = (seq_len + 31) / 32
    
    // K/V tile stride (32 rows × 128 cols × 1 byte = 4096 bytes)
    s_mov_b32 s26, 4096
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE
    // ========================================================================
    v_mov_b32_e32 v70, s3                  // running_max = -inf
    v_mov_b32_e32 v71, 0                   // running_sum = 0
    
    // Initialize O accumulator (v80-v95 for first HD tile only for now)
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // SETUP BUFFER DESCRIPTORS (4 SGPRs each: base_lo, base_hi, size, flags)
    // ========================================================================
    // Q descriptor at s[8:11]: s[8:9] already has Q pointer
    s_mov_b32 s10, -1              // size = max
    s_mov_b32 s11, 0x20000         // flags
    
    // K descriptor at s[12:15]: s[12:13] already has K pointer
    s_mov_b32 s14, -1              // size = max
    s_mov_b32 s15, 0x20000         // flags
    
    // V descriptor at s[16:19]: s[16:17] already has V pointer
    s_mov_b32 s18, -1              // size = max
    s_mov_b32 s19, 0x20000         // flags
    
    // ========================================================================
    // LOAD Q TO LDS (stays for all tiles)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v0
    
    s_mov_b32 m0, 0
    s_mov_b32 s20, 0
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
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // K-tile offset and counter
    s_mov_b32 s27, 0                       // k_offset = 0
    s_mov_b32 s28, 0                       // tile_idx = 0
    
    // K and V buffer descriptors already set up above with size=-1 (max)
    // DO NOT overwrite s14, s18 - they need to allow access beyond 4096 bytes
    // for multi-tile access with s27 offset

    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
K_TILE_LOOP:
    // Recalculate v1 for K load (might have been clobbered)
    v_lshlrev_b32_e32 v1, 4, v0
    
    // Load K tile to LDS at offset 4096 (matching working kernel pattern)
    s_mov_b32 m0, 4096
    s_mov_b32 s20, s27
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 5120
    s_add_i32 s20, s27, 1024
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 6144
    s_add_i32 s20, s27, 2048
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 7168
    s_add_i32 s20, s27, 3072
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // QK MFMA: S^T[32×32] = K @ Q^T
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
    
    // Read Q and K from LDS (same pattern as working kernel)
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4
    v_add_u32_e32 v6, 4096, v5            // K base at LDS offset 4096
    
    // 8 QK MFMA passes: S^T = K @ Q^T
    // A operand = K, B operand = Q (swapped for transpose)
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
    // ONLINE SOFTMAX UPDATE
    // ========================================================================
    
    // Find tile_max
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
    
    // Cross-lane max
    v_mov_b32_e32 v20, v21
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21           // v21 = tile_max
    
    // new_max = max(running_max, tile_max)
    v_max_f32_e32 v22, v70, v21           // v22 = new_max
    
    // correction = exp((running_max - new_max) * scale)
    v_sub_f32_e32 v23, v70, v22
    v_mul_f32_e32 v23, s2, v23
    v_exp_f32_e32 v23, v23                // v23 = correction
    s_nop 3
    
    // Rescale O accumulator: O = O * correction
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v\i, v23
    .endr
    
    // Rescale running_sum
    v_mul_f32_e32 v71, v71, v23
    
    // Update running_max
    v_mov_b32_e32 v70, v22
    
    // Compute P = exp((S - new_max) * scale)
    // MUST use new_max (v22), not tile_max (v21) for correct online softmax!
    v_mul_f32_e32 v23, s2, v22            // new_max * scale
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
    
    // Compute tile_sum
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
    
    v_mov_b32_e32 v22, v20
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v22, v20
    v_add_f32_e32 v20, v20, v22           // v20 = tile_sum
    
    // Update running_sum (P is NOT normalized per-tile for online softmax)
    v_add_f32_e32 v71, v71, v20
    
    // ========================================================================
    // STORE P TO LDS, LOAD V
    // ========================================================================
    s_barrier
    
    // Store P (transposed) to LDS offset 4096 (K is no longer needed)
    // Q stays at offset 0 for next K-tile
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v3, 2, v3
    
    .macro STORE_P vreg, key_mod4, key_8_group
        v_mov_b32_e32 v7, \key_mod4
        v_add_u32_e32 v7, v7, v3
        v_add_u32_e32 v7, \key_8_group * 8, v7
        v_lshlrev_b32_e32 v8, 5, v2
        v_add_u32_e32 v8, v8, v7
        v_lshlrev_b32_e32 v8, 2, v8
        v_add_u32_e32 v8, 4096, v8       // Store at offset 4096
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
    
    s_waitcnt lgkmcnt(0)              // Wait for P stores to complete
    s_barrier                          // Sync all threads
    
    // Load V tile to LDS at offset 8192 (Q at 0, P at 4096)
    // MUST recalculate v1 - same issue as K load!
    v_lshlrev_b32_e32 v1, 4, v0
    s_mov_b32 m0, 8192
    buffer_load_dwordx4 v1, s[16:19], s27 offen lds
    s_add_i32 s20, s27, 1024
    s_mov_b32 m0, 9216
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    s_add_i32 s20, s27, 2048
    s_mov_b32 m0, 10240
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    s_add_i32 s20, s27, 3072
    s_mov_b32 m0, 11264
    buffer_load_dwordx4 v1, s[16:19], s20 offen lds
    
    s_waitcnt lgkmcnt(0) vmcnt(0)
    s_barrier
    
    // ========================================================================
    // PV MFMA: O += P @ V (first HD tile only)
    // P at offset 4096, V at offset 8192
    // ========================================================================
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    
    // K-pass 0 (k=0..15): Read P from offset 4096
    v_lshlrev_b32_e32 v4, 7, v2
    v_lshlrev_b32_e32 v5, 5, v3
    v_add_u32_e32 v4, v4, v5
    v_add_u32_e32 v4, 4096, v4            // P is at offset 4096
    
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
    
    // V read from LDS offset 8192
    v_lshlrev_b32_e32 v5, 10, v3
    v_add_u32_e32 v5, v5, v2
    v_add_u32_e32 v5, 8192, v5            // V is at offset 8192
    
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
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[30:31], v[80:95]
    s_nop 15
    
    // K-pass 1 (k=16..31): P at offset 4096 + 64
    v_add_u32_e32 v4, 64, v4              // Advance P read address
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
    v_add_u32_e32 v5, 8192 + 2048, v5     // V at offset 8192 + 2048 for second half
    
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
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[30:31], v[80:95]
    s_nop 15
    
    // Advance to next K/V tile
    s_add_i32 s27, s27, s26               // k_offset += k_stride
    s_add_i32 s28, s28, 1                 // tile_idx++
    s_cmp_lt_i32 s28, s25                 // if tile_idx < num_tiles
    s_cbranch_scc1 K_TILE_LOOP
    
    // ========================================================================
    // FINAL NORMALIZATION: O = O / running_sum
    // ========================================================================
    v_rcp_f32_e32 v72, v71
    s_nop 3
    
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v\i, v72
    .endr
    
    // ========================================================================
    // STORE OUTPUT (first 32 columns only)
    // ========================================================================
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v3, 2, v3
    
    .macro STORE_O vreg, row_mod4, row_8_group
        v_mov_b32_e32 v7, \row_mod4
        v_add_u32_e32 v7, v7, v3
        v_add_u32_e32 v7, \row_8_group * 8, v7
        v_lshlrev_b32_e32 v7, 7, v7           // row * 128 (HD)
        v_add_u32_e32 v7, v7, v2              // + col (within first 32)
        v_lshlrev_b32_e32 v7, 2, v7           // byte offset
        v_mov_b32_e32 v10, s4
        v_mov_b32_e32 v11, s5
        v_add_co_u32_e32 v10, vcc, v7, v10
        v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
        flat_store_dword v[10:11], \vreg
    .endm
    
    STORE_O v80, 0, 0
    STORE_O v81, 1, 0
    STORE_O v82, 2, 0
    STORE_O v83, 3, 0
    STORE_O v84, 0, 1
    STORE_O v85, 1, 1
    STORE_O v86, 2, 1
    STORE_O v87, 3, 1
    STORE_O v88, 0, 2
    STORE_O v89, 1, 2
    STORE_O v90, 2, 2
    STORE_O v91, 3, 2
    STORE_O v92, 0, 3
    STORE_O v93, 1, 3
    STORE_O v94, 2, 3
    STORE_O v95, 3, 3
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13fwd_fp8_kloopE
    .amdhsa_group_segment_fixed_size 12288
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 100
    .amdhsa_next_free_sgpr 32
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
  - .name: _ZN5aiter13fwd_fp8_kloopE
    .symbol: _ZN5aiter13fwd_fp8_kloopE.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 100
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

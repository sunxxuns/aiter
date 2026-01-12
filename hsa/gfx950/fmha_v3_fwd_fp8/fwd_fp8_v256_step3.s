// FP8 Flash Attention 256T - Step 3: QK MFMA
// Each wave (64 threads) runs full 32×32 MFMA independently
// Wave stores only its 8 output rows
//
// This is NOT optimal (4x redundant computation) but validates structure.
// Optimization in Phase 7+.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16fwd_fp8_v256_s3E
.p2align 8
.type _ZN5aiter16fwd_fp8_v256_s3E,@function

_ZN5aiter16fwd_fp8_v256_s3E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // THREAD ID: 256 threads = 4 waves × 64 lanes
    // ========================================================================
    v_lshrrev_b32_e32 v1, 6, v0           // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0              // v0 = lane_id (0-63)
    
    // ========================================================================
    // SAVE WORKGROUP ID & LOAD ARGS
    // ========================================================================
    s_mov_b32 s31, s2                     // q_tile_idx
    
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // CONSTANTS
    // ========================================================================
    // Scale = 1/sqrt(128) * log2(e) = 0.08838835 * 1.442695 ≈ 0.1275
    s_mov_b32 s2, 0x3e028442              // scale factor
    
    // ========================================================================
    // COMPUTE Q-TILE OFFSETS
    // ========================================================================
    s_lshl_b32 s29, s31, 12               // Q offset = tile * 4096 (FP8)
    s_lshl_b32 s30, s31, 14               // O offset = tile * 16384 (F32)
    
    s_add_u32 s8, s8, s29
    s_addc_u32 s9, s9, 0
    s_add_u32 s4, s4, s30
    s_addc_u32 s5, s5, 0
    
    // ========================================================================
    // BUFFER DESCRIPTORS
    // ========================================================================
    // Q buffer: s[8:11]
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    
    // K buffer: s[12:15]
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    
    // V buffer: s[16:19]
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000
    
    // O buffer: s[4:7]
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // ========================================================================
    // INIT ONLINE SOFTMAX STATE
    // ========================================================================
    v_mov_b32_e32 v70, 0xff800000         // running_max = -inf
    v_mov_b32_e32 v71, 0                  // running_sum = 0
    
    // Init O accumulator (4 HD tiles × 16 regs = 64 regs: v80-v143)
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // LOAD Q TO LDS (all 256 threads cooperate)
    // ========================================================================
    // Q tile: 32 rows × 128 cols × 1 byte = 4096 bytes
    // 256 threads × 16 bytes/thread = 4096 bytes ✓
    
    v_lshlrev_b32_e32 v2, 6, v1           // wave * 64
    v_add_u32_e32 v2, v0, v2              // thread_id (0-255)
    v_lshlrev_b32_e32 v2, 4, v2           // thread_id * 16 (LDS offset)
    
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v2, s[8:11], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // K-TILE LOOP SETUP
    // ========================================================================
    s_mov_b32 s27, 0                       // k_offset = 0
    s_lshr_b32 s28, s24, 5                 // num_k_tiles = seq_len / 32
    s_mov_b32 s26, 4096                    // k_stride = 32 * 128 * 1 = 4096 bytes
    
    // ========================================================================
    // K-TILE LOOP START
    // ========================================================================
K_TILE_LOOP:
    // ------------------------------------------------------------
    // Load K tile to LDS (offset 4096)
    // buffer_load ... offen lds: LDS addr = m0 + vgpr
    // ------------------------------------------------------------
    v_lshlrev_b32_e32 v2, 6, v1           // wave * 64
    v_add_u32_e32 v2, v0, v2              // thread_id (0-255)
    v_lshlrev_b32_e32 v2, 4, v2           // thread_id * 16 (per-thread offset)
    
    s_mov_b32 m0, 4096                    // LDS base offset for K
    buffer_load_dwordx4 v2, s[12:15], s27 offen lds
    
    // Load V tile to LDS (offset 8192)
    // Reuse v2 (same per-thread offset, different LDS base)
    s_mov_b32 m0, 8192                    // LDS base offset for V
    buffer_load_dwordx4 v2, s[16:19], s27 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // Only wave 0 computes (avoid race conditions on LDS P storage)
    // ========================================================================
    v_cmp_eq_u32_e32 vcc, 0, v1           // wave_id == 0?
    s_and_saveexec_b64 s[22:23], vcc
    s_cbranch_execz K_TILE_END            // Skip if not wave 0
    
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
    
    // Compute lane addresses in LDS
    v_and_b32_e32 v2, 31, v0              // lane % 32
    v_lshrrev_b32_e32 v3, 5, v0           // lane / 32 (0 or 1)
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128 (Q is 128 cols)
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4              // Q LDS base
    v_add_u32_e32 v6, 4096, v5            // K LDS base (K at offset 4096)
    
    // 8 MFMA passes over HD (128 / 16 = 8 passes)
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \k_off, v6      // K offset
        v_add_u32_e32 v8, \k_off, v5      // Q offset
        ds_read_b64 v[20:21], v7          // Read K
        ds_read_b64 v[22:23], v8          // Read Q
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v20       // K → AGPR
        v_accvgpr_write_b32 a1, v21
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
        s_nop 15
    .endr
    s_nop 15
    
    // ========================================================================
    // ONLINE SOFTMAX
    // ========================================================================
    
    // Find tile_max from S (v32-v47)
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
    
    // Cross-lane max reduction (swap lanes 0-31 with 32-63)
    v_mov_b32_e32 v22, v21
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v22, v21
    v_max_f32_e32 v21, v21, v22
    
    // tile_max = v21
    // new_max = max(running_max, tile_max)
    v_max_f32_e32 v22, v70, v21
    
    // Correction factor: correction = exp((old_max - new_max) * scale)
    v_sub_f32_e32 v23, v70, v22           // old_max - new_max
    v_mul_f32_e32 v23, s2, v23            // * scale
    v_exp_f32_e32 v23, v23                // correction
    
    // Rescale O accumulator
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    .irp i, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    .irp i, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    .irp i, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    v_mul_f32_e32 v71, v23, v71           // Rescale running_sum
    
    // Update running_max
    v_mov_b32_e32 v70, v22
    
    // Compute P = exp((S - new_max) * scale)
    .irp i, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
        v_sub_f32_e32 v\i, v\i, v22       // S - new_max
        v_mul_f32_e32 v\i, s2, v\i        // * scale
        v_exp_f32_e32 v\i, v\i            // P = exp(...)
    .endr
    
    // Sum P for tile_sum
    v_add_f32_e32 v24, v32, v33
    v_add_f32_e32 v24, v24, v34
    v_add_f32_e32 v24, v24, v35
    v_add_f32_e32 v24, v24, v36
    v_add_f32_e32 v24, v24, v37
    v_add_f32_e32 v24, v24, v38
    v_add_f32_e32 v24, v24, v39
    v_add_f32_e32 v24, v24, v40
    v_add_f32_e32 v24, v24, v41
    v_add_f32_e32 v24, v24, v42
    v_add_f32_e32 v24, v24, v43
    v_add_f32_e32 v24, v24, v44
    v_add_f32_e32 v24, v24, v45
    v_add_f32_e32 v24, v24, v46
    v_add_f32_e32 v24, v24, v47
    
    // Cross-lane sum (swap lanes 0-31 with 32-63)
    v_mov_b32_e32 v25, v24
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v25, v24
    v_add_f32_e32 v24, v24, v25
    
    // Update running_sum
    v_add_f32_e32 v71, v71, v24
    
    // ========================================================================
    // P → FP8 → LDS
    // ========================================================================
    // Store P to LDS at offset 12288 (after V)
    // P is 32×32 F32, convert to FP8 for MFMA
    
    // Recalculate v2, v3 (clobbered during K load)
    v_and_b32_e32 v2, 31, v0              // lane % 32
    v_lshrrev_b32_e32 v3, 5, v0           // lane / 32
    
    v_cvt_pk_fp8_f32 v28, v32, v33
    v_cvt_pk_fp8_f32 v29, v34, v35
    v_lshlrev_b32_e32 v25, 7, v2          // row * 128
    v_lshlrev_b32_e32 v26, 3, v3          // half * 8
    v_add_u32_e32 v25, v25, v26
    v_add_u32_e32 v25, 12288, v25         // P LDS offset
    ds_write_b64 v25, v[28:29]
    
    v_cvt_pk_fp8_f32 v28, v36, v37
    v_cvt_pk_fp8_f32 v29, v38, v39
    v_add_u32_e32 v26, 16, v25
    ds_write_b64 v26, v[28:29]
    
    v_cvt_pk_fp8_f32 v28, v40, v41
    v_cvt_pk_fp8_f32 v29, v42, v43
    v_add_u32_e32 v26, 32, v25
    ds_write_b64 v26, v[28:29]
    
    v_cvt_pk_fp8_f32 v28, v44, v45
    v_cvt_pk_fp8_f32 v29, v46, v47
    v_add_u32_e32 v26, 48, v25
    ds_write_b64 v26, v[28:29]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // PV MFMA: O += P @ V (4 HD tiles)
    // ========================================================================
    // V is at LDS offset 8192, P is at 12288
    // Each HD tile: 32 columns of V
    
    // HD Tile 0 (V cols 0-31)
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4              // P base
    v_add_u32_e32 v5, 12288, v5           // P LDS offset
    v_add_u32_e32 v6, 8192, v4            // V base (8192 + half*8)
    v_lshlrev_b32_e32 v7, 7, v2           // row * 128 for V row offset
    v_add_u32_e32 v6, v6, v7
    
    .irp pv_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \pv_off, v5     // P offset (row index in P)
        ds_read_b64 v[28:29], v7          // Read P
        v_add_u32_e32 v8, \pv_off, v6     // V offset (col index in V)
        ds_read_b64 v[30:31], v8          // Read V (first 32 cols)
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v28
        v_accvgpr_write_b32 a1, v29
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[30:31], v[80:95]
        s_nop 15
    .endr
    s_nop 15
    
    // HD Tile 1 (V cols 32-63)
    .irp pv_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \pv_off, v5
        ds_read_b64 v[28:29], v7
        v_add_u32_e32 v8, \pv_off + 32, v6   // V offset +32 for cols 32-63
        ds_read_b64 v[30:31], v8
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v28
        v_accvgpr_write_b32 a1, v29
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[96:111], a[0:1], v[30:31], v[96:111]
        s_nop 15
    .endr
    s_nop 15
    
    // HD Tile 2 (V cols 64-95)
    .irp pv_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \pv_off, v5
        ds_read_b64 v[28:29], v7
        v_add_u32_e32 v8, \pv_off + 64, v6
        ds_read_b64 v[30:31], v8
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v28
        v_accvgpr_write_b32 a1, v29
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[112:127], a[0:1], v[30:31], v[112:127]
        s_nop 15
    .endr
    s_nop 15
    
    // HD Tile 3 (V cols 96-127)
    .irp pv_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \pv_off, v5
        ds_read_b64 v[28:29], v7
        v_add_u32_e32 v8, \pv_off + 96, v6
        ds_read_b64 v[30:31], v8
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v28
        v_accvgpr_write_b32 a1, v29
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[128:143], a[0:1], v[30:31], v[128:143]
        s_nop 15
    .endr
    s_nop 15
    
K_TILE_END:
    // Restore exec for all waves
    s_mov_b64 exec, -1
    
    // ========================================================================
    // K-TILE LOOP END
    // ========================================================================
    s_add_i32 s27, s27, s26               // k_offset += k_stride
    s_sub_i32 s28, s28, 1                 // num_tiles--
    s_cmp_gt_i32 s28, 0
    s_cbranch_scc1 K_TILE_LOOP
    
    // ========================================================================
    // FINAL OUTPUT: O = O / running_sum
    // Each wave stores only its 8 rows
    // ========================================================================
    
    // Compute which 8 rows this wave owns
    // Wave 0: rows 0-7, Wave 1: rows 8-15, etc.
    // MFMA output mapping: lane determines output position
    
    v_rcp_f32_e32 v72, v71                // 1 / running_sum
    
    // Normalize O
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v72, v\i
    .endr
    .irp i, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        v_mul_f32_e32 v\i, v72, v\i
    .endr
    .irp i, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        v_mul_f32_e32 v\i, v72, v\i
    .endr
    .irp i, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
        v_mul_f32_e32 v\i, v72, v\i
    .endr
    
    // ========================================================================
    // STORE OUTPUT (wave 0 only - exec already masked above)
    // ========================================================================
    
    // MFMA output layout: each lane holds 16 output elements
    // lane % 32 = column index (0-31)
    // lane / 32 = row group (0 = rows 0-15, 1 = rows 16-31)
    
    v_and_b32_e32 v50, 31, v0             // col = lane % 32
    v_lshrrev_b32_e32 v51, 5, v0          // half = lane / 32 (0 or 1)
    v_lshlrev_b32_e32 v52, 4, v51         // row_base = half * 16
    
    // Macro for store: row, vreg, hd_col_offset
    .macro STORE_HD vreg, row_idx, hd_off
        v_add_u32_e32 v53, \row_idx, v52  // row = row_base + row_idx
        v_lshlrev_b32_e32 v54, 9, v53     // row * 512 (= row * 128 * 4)
        v_add_u32_e32 v55, \hd_off, v50   // col + hd_offset
        v_lshlrev_b32_e32 v55, 2, v55     // col_bytes
        v_add_u32_e32 v54, v54, v55       // total offset
        buffer_store_dword \vreg, v54, s[4:7], 0 offen
    .endm
    
    // HD Tile 0 (cols 0-31): v80-v95
    STORE_HD v80, 0, 0
    STORE_HD v81, 1, 0
    STORE_HD v82, 2, 0
    STORE_HD v83, 3, 0
    STORE_HD v84, 4, 0
    STORE_HD v85, 5, 0
    STORE_HD v86, 6, 0
    STORE_HD v87, 7, 0
    STORE_HD v88, 8, 0
    STORE_HD v89, 9, 0
    STORE_HD v90, 10, 0
    STORE_HD v91, 11, 0
    STORE_HD v92, 12, 0
    STORE_HD v93, 13, 0
    STORE_HD v94, 14, 0
    STORE_HD v95, 15, 0
    
    // HD Tile 1 (cols 32-63): v96-v111
    STORE_HD v96, 0, 32
    STORE_HD v97, 1, 32
    STORE_HD v98, 2, 32
    STORE_HD v99, 3, 32
    STORE_HD v100, 4, 32
    STORE_HD v101, 5, 32
    STORE_HD v102, 6, 32
    STORE_HD v103, 7, 32
    STORE_HD v104, 8, 32
    STORE_HD v105, 9, 32
    STORE_HD v106, 10, 32
    STORE_HD v107, 11, 32
    STORE_HD v108, 12, 32
    STORE_HD v109, 13, 32
    STORE_HD v110, 14, 32
    STORE_HD v111, 15, 32
    
    // HD Tile 2 (cols 64-95): v112-v127
    STORE_HD v112, 0, 64
    STORE_HD v113, 1, 64
    STORE_HD v114, 2, 64
    STORE_HD v115, 3, 64
    STORE_HD v116, 4, 64
    STORE_HD v117, 5, 64
    STORE_HD v118, 6, 64
    STORE_HD v119, 7, 64
    STORE_HD v120, 8, 64
    STORE_HD v121, 9, 64
    STORE_HD v122, 10, 64
    STORE_HD v123, 11, 64
    STORE_HD v124, 12, 64
    STORE_HD v125, 13, 64
    STORE_HD v126, 14, 64
    STORE_HD v127, 15, 64
    
    // HD Tile 3 (cols 96-127): v128-v143
    STORE_HD v128, 0, 96
    STORE_HD v129, 1, 96
    STORE_HD v130, 2, 96
    STORE_HD v131, 3, 96
    STORE_HD v132, 4, 96
    STORE_HD v133, 5, 96
    STORE_HD v134, 6, 96
    STORE_HD v135, 7, 96
    STORE_HD v136, 8, 96
    STORE_HD v137, 9, 96
    STORE_HD v138, 10, 96
    STORE_HD v139, 11, 96
    STORE_HD v140, 12, 96
    STORE_HD v141, 13, 96
    STORE_HD v142, 14, 96
    STORE_HD v143, 15, 96

SKIP_STORE:
    s_mov_b64 exec, -1                    // Restore exec mask
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_v256_s3E
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 156
    .amdhsa_next_free_sgpr 40
    .amdhsa_accum_offset 156
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
  - .name: _ZN5aiter16fwd_fp8_v256_s3E
    .symbol: _ZN5aiter16fwd_fp8_v256_s3E.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 40
    .vgpr_count: 156
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

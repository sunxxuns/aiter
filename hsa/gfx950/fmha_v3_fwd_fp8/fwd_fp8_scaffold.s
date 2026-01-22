// FP8 QK+PV scaffold kernel (no softmax) for perf validation
// - Focus: LDS layout + bulk loads/stores + MFMA throughput
// - NOT numerically correct (no softmax, P reuse)
// Grid: (num_q_blocks, batch * heads, 1)
// Each block: 256 threads (4 waves), processes 128 Q rows

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set Q_LDS, 0                 // Q tile 0
.set Q_LDS1, 16896            // Q tile 1 (128×132)
.set K_LDS0, 33792            // 32×128 (row-major)
.set K_LDS1, 37888            // ping-pong
.set V_LDS0, 41984            // 32×128 (row-major, contiguous with V_LDS1)
.set V_LDS1, 46080            // ping-pong (V_LDS0 + 4096)
.set LDS_SIZE, 50176          // aligned

.text
.globl _fwd_fp8_scaffold
.p2align 8
.type _fwd_fp8_scaffold,@function

_fwd_fp8_scaffold:
    s_mov_b64 exec, -1

    // ------------------------------------------------------------------------
    // Load kernel arguments
    // ------------------------------------------------------------------------
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr
    s_load_dwordx2 s[16:17], s[0:1], 24   // V_ptr
    s_load_dword s20, s[0:1], 32          // num_k_tiles
    s_load_dword s21, s[0:1], 36          // stride_qh
    s_load_dword s22, s[0:1], 40          // stride_kh
    s_load_dword s23, s[0:1], 44          // stride_vh
    s_load_dword s24, s[0:1], 48          // stride_oh (bytes)
    s_waitcnt lgkmcnt(0)

    // Buffer descriptors (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000

    // ------------------------------------------------------------------------
    // Compute block offsets
    // ------------------------------------------------------------------------
    // Q offset = head * stride_qh + q_block_base * 16384
    // q_block_base = workgroup_id_x * 2
    s_mul_i32 s26, s3, s21
    s_lshl_b32 s32, s2, 15               // * 32768
    s_add_u32 s26, s26, s32

    // K/V offsets = head * stride_kh/vh
    s_mul_i32 s27, s3, s22
    s_mul_i32 s28, s3, s23
    // Fold head offsets into K/V base pointers for buffer_load...lds
    s_add_u32 s12, s12, s27
    s_addc_u32 s13, s13, 0
    s_add_u32 s16, s16, s28
    s_addc_u32 s17, s17, 0

    // LDS base constants for ping-pong buffers
    s_mov_b32 s40, K_LDS0
    s_mov_b32 s41, K_LDS1
    s_mov_b32 s42, V_LDS0
    s_mov_b32 s43, V_LDS1

    // O offset = head * stride_oh + (q_block_base) * 65536
    // q_block_base = workgroup_id_x * 2
    s_mul_i32 s29, s3, s24
    s_lshl_b32 s33, s2, 17               // * 131072
    s_add_u32 s29, s29, s33

    // Bitop3 constants for V swizzle/read
    s_movk_i32 s24, 0x70
    s_movk_i32 s25, 0xb80

    // ------------------------------------------------------------------------
    // Thread indexing
    // ------------------------------------------------------------------------
    v_mov_b32_e32 v60, v0                // tid (0-511), keep for stores
    v_lshrrev_b32_e32 v9, 6, v60         // wave_id = tid / 64 (0-7)
    v_and_b32_e32 v10, 63, v60           // lane_id = tid % 64
    v_and_b32_e32 v61, 0xFF, v60
    v_lshlrev_b32_e32 v61, 4, v61        // (tid & 255) * 16 (prefetch vaddr)

    // ------------------------------------------------------------------------
    // Load Q tile to LDS (pitch=132)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v1, 6, v60         // tid * 64 bytes
    v_add_u32_e32 v2, s26, v1            // Q global offset

    // Q LDS address: row = tid >> 1, col = (tid & 1) * 64
    v_and_b32_e32 v11, 0xFF, v60         // tid_in_tile (0-255)
    v_lshrrev_b32_e32 v11, 1, v11        // row
    v_lshlrev_b32_e32 v12, 7, v11
    v_lshlrev_b32_e32 v11, 2, v11
    v_add_u32_e32 v12, v12, v11          // row * 132
    v_and_b32_e32 v11, 1, v60
    v_lshlrev_b32_e32 v11, 6, v11
    v_add_u32_e32 v12, v12, v11
    // tile offset = (tid >> 8) * 16896 = (tid>>8)<<14 + (tid>>8)<<9
    v_lshrrev_b32_e32 v21, 8, v60
    v_lshlrev_b32_e32 v22, 14, v21
    v_lshlrev_b32_e32 v23, 9, v21
    v_add_u32_e32 v22, v22, v23
    v_add_u32_e32 v20, v22, v12

    buffer_load_dwordx4 v[32:35], v2, s[8:11], 0 offen
    v_add_u32_e32 v3, 16, v2
    buffer_load_dwordx4 v[36:39], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 32, v2
    buffer_load_dwordx4 v[40:43], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 48, v2
    buffer_load_dwordx4 v[44:47], v3, s[8:11], 0 offen
    s_waitcnt vmcnt(0)

    ds_write_b128 v20, v[32:35]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[36:39]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[40:43]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[44:47]

    s_waitcnt lgkmcnt(0)

    // ------------------------------------------------------------------------
    // Compute MFMA LDS read addresses (pitch=132)
    // ------------------------------------------------------------------------
    v_and_b32_e32 v11, 15, v10           // lane & 15
    v_lshrrev_b32_e32 v12, 4, v10
    v_and_b32_e32 v12, 1, v12
    v_lshlrev_b32_e32 v12, 4, v12
    v_add_u32_e32 v13, v11, v12          // mfma_row

    v_lshlrev_b32_e32 v14, 7, v13        // row * 128
    v_lshlrev_b32_e32 v15, 2, v13        // row * 4
    v_add_u32_e32 v14, v14, v15          // Q row offset (132)
    v_lshlrev_b32_e32 v18, 7, v13        // K/V row offset (128)

    v_cmp_ge_u32_e64 vcc, v10, 32
    v_cndmask_b32_e64 v11, 0, 16, vcc    // k_off1
    v_cndmask_b32_e64 v12, 32, 48, vcc   // k_off2

    // wave_in_tile = wave_id & 3, tile_id = wave_id >> 2
    v_and_b32_e32 v15, 3, v9
    v_lshrrev_b32_e32 v16, 2, v9
    // wave_in_tile * 4224 = (wave*4096 + wave*128)
    v_lshlrev_b32_e32 v17, 12, v15
    v_lshlrev_b32_e32 v19, 7, v15
    v_add_u32_e32 v17, v17, v19
    // tile_id * 16896 = (tile<<14) + (tile<<9)
    v_lshlrev_b32_e32 v19, 14, v16
    v_lshlrev_b32_e32 v20, 9, v16
    v_add_u32_e32 v19, v19, v20
    v_add_u32_e32 v17, v17, v19          // Q base

    v_add_u32_e32 v58, v17, v14          // Q base + row
    v_add_u32_e32 v24, v58, v11          // Q addr1 base
    v_add_u32_e32 v25, v58, v12          // Q addr2 base

    // ------------------------------------------------------------------------
    // Initialize O accumulators (v[64:127])
    // ------------------------------------------------------------------------
    .irp i, 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
        v_mov_b32_e32 v\i, 0
    .endr

    // ------------------------------------------------------------------------
    // K-tile loop: QK + PV (no softmax)
    // ------------------------------------------------------------------------
    v_mov_b32_e32 v59, 0x05040100        // selector for v_perm_b32
    s_mov_b32 s30, 0                     // tile_idx (pair)
    s_mov_b32 s31, 0                     // K/V tile soffset

    // Preload tiles 0/1 into K_LDS0/1 and swizzled V (pair)
    v_mov_b32_e32 v2, 256
    v_cmp_lt_u32_e32 vcc, v60, v2
    s_and_saveexec_b64 s[22:23], vcc
    v_add_u32_e32 v13, s31, v61          // K global offset (tile0)
    buffer_load_dwordx4 v[20:23], v13, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v14, s40, v61          // K_LDS0 + (tid&255)*16
    ds_write_b128 v14, v[20:23]

    s_add_u32 s34, s31, 4096
    v_add_u32_e32 v13, s34, v61          // K global offset (tile1)
    buffer_load_dwordx4 v[20:23], v13, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v14, s41, v61          // K_LDS1 + (tid&255)*16
    ds_write_b128 v14, v[20:23]
    s_mov_b64 exec, s[22:23]

    // Swizzled V load: 16B for row r and row r+32
    v_mov_b32_e32 v2, 256
    v_cmp_lt_u32_e32 vcc, v60, v2
    s_and_saveexec_b64 s[22:23], vcc
    v_lshrrev_b32_e32 v4, 3, v60          // row = tid >> 3 (0..31)
    v_and_b32_e32 v5, 7, v60              // col_block = tid & 7 (0..7)
    v_lshlrev_b32_e32 v4, 7, v4           // row * 128
    v_lshlrev_b32_e32 v5, 4, v5           // col_block * 16
    v_add_u32_e32 v2, v4, v5              // byte offset within V
    buffer_load_dwordx4 v[40:43], v2, s[16:19], s31 offen
    v_add_u32_e32 v3, 4096, v2            // row + 32
    buffer_load_dwordx4 v[44:47], v3, s[16:19], s31 offen

    // Triton-style LDS write swizzle (bitop3:0x78)
    s_movk_i32 s26, 0x70
    v_lshlrev_b32_e32 v4, 4, v60          // tid * 16 bytes
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x78
    v_add_u32_e32 v4, s42, v4
    s_waitcnt vmcnt(0)
    ds_write_b128 v4, v[40:43]
    ds_write_b128 v4, v[44:47] offset:4096
    s_mov_b64 exec, s[22:23]
    s_waitcnt lgkmcnt(0)
    s_waitcnt vmcnt(0)

K_LOOP:
    s_waitcnt vmcnt(0)
    s_barrier
    s_cmp_ge_u32 s30, s20
    s_cbranch_scc1 K_LOOP_END

    // Tile pair bases (V_LDS0 holds swizzled K=64 rows)
    v_mov_b32_e32 v29, s40                      // K tile0 base
    v_mov_b32_e32 v31, s41                      // K tile1 base
    v_mov_b32_e32 v56, s42                      // V base (tile0)

    // Compute Q/K read addresses for tile 0
    v_add_u32_e32 v26, v29, v18                 // K base + row
    v_add_u32_e32 v27, v26, v11                 // K addr1
    v_add_u32_e32 v28, v26, v12                 // K addr2

    // QK MFMA (K=64 × 2) for tile 0
    .irp i, 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
        v_mov_b32_e32 v\i, 0
    .endr
    ds_read_b128 v[0:3], v24
    ds_read_b128 v[4:7], v25
    ds_read_b128 v[16:19], v27
    ds_read_b128 v[20:23], v28
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    ds_read_b128 v[0:3], v24 offset:64
    ds_read_b128 v[4:7], v25 offset:64
    ds_read_b128 v[16:19], v27 offset:64
    ds_read_b128 v[20:23], v28 offset:64
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    // Pack P0 (tile 0) to FP8 (v48-v51) - Triton-style
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v48, v32, v33, 1.0
    v_cvt_scalef32_pk_fp8_f32 v48, v34, v35, 1.0 op_sel:[0,0,0,1]
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v49, v36, v37, 1.0
    v_cvt_scalef32_pk_fp8_f32 v49, v38, v39, 1.0 op_sel:[0,0,0,1]
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v50, v40, v41, 1.0
    v_cvt_scalef32_pk_fp8_f32 v50, v42, v43, 1.0 op_sel:[0,0,0,1]
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v51, v44, v45, 1.0
    v_cvt_scalef32_pk_fp8_f32 v51, v46, v47, 1.0 op_sel:[0,0,0,1]

    // Compute Q/K read addresses for tile 1
    v_add_u32_e32 v26, v31, v18                 // K base + row
    v_add_u32_e32 v27, v26, v11                 // K addr1
    v_add_u32_e32 v28, v26, v12                 // K addr2

    // QK MFMA (K=64 × 2) for tile 1
    .irp i, 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
        v_mov_b32_e32 v\i, 0
    .endr
    ds_read_b128 v[0:3], v24
    ds_read_b128 v[4:7], v25
    ds_read_b128 v[16:19], v27
    ds_read_b128 v[20:23], v28
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    ds_read_b128 v[0:3], v24 offset:64
    ds_read_b128 v[4:7], v25 offset:64
    ds_read_b128 v[16:19], v27 offset:64
    ds_read_b128 v[20:23], v28 offset:64
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[0:7], v[16:23], v[32:47]

    // Pack P1 (tile 1) to FP8 (v52-v55) - Triton-style
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v52, v32, v33, 1.0
    v_cvt_scalef32_pk_fp8_f32 v52, v34, v35, 1.0 op_sel:[0,0,0,1]
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v53, v36, v37, 1.0
    v_cvt_scalef32_pk_fp8_f32 v53, v38, v39, 1.0 op_sel:[0,0,0,1]
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v54, v40, v41, 1.0
    v_cvt_scalef32_pk_fp8_f32 v54, v42, v43, 1.0 op_sel:[0,0,0,1]
    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
    v_cvt_scalef32_pk_fp8_f32 v55, v44, v45, 1.0
    v_cvt_scalef32_pk_fp8_f32 v55, v46, v47, 1.0 op_sel:[0,0,0,1]

    // Build A operand from packed P (byte-level transpose)
    // row = (lane & 15) + 16 * ((lane >> 4) & 1)
    v_and_b32_e32 v57, 15, v10
    v_and_b32_e32 v58, 16, v10
    v_add_u32_e32 v57, v57, v58            // row

    // src_half = (row & 4) * 8  (0 or 32)
    v_and_b32_e32 v59, 4, v57
    v_lshlrev_b32_e32 v56, 3, v59

    // byte_shift = (row & 3) * 8
    v_and_b32_e32 v59, 3, v57
    v_lshlrev_b32_e32 v63, 3, v59

    // group = (row & ~4) >> 3
    v_and_b32_e32 v62, 0xFFFFFFFB, v57
    v_lshrrev_b32_e32 v62, 3, v62

    // Select packed source word for tile0 (v32) and tile1 (v33)
    v_mov_b32_e32 v32, v48
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v32, v32, v49, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v32, v32, v50, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v32, v32, v51, vcc

    v_mov_b32_e32 v33, v52
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v33, v33, v53, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v33, v33, v54, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v33, v33, v55, vcc

    // col_base = (lane & 32) >> 1  (0 or 16)
    v_and_b32_e32 v58, 32, v10
    v_lshrrev_b32_e32 v58, 1, v58

    // Group 0 -> v48 (cols 0..3)
    v_add_u32_e32 v4, v58, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v33
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v48, v0

    // Group 1 -> v49 (cols 4..7)
    v_add_u32_e32 v4, v58, v56
    v_add_u32_e32 v4, 4, v4
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v33
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v49, v0

    // Group 2 -> v50 (cols 8..11)
    v_add_u32_e32 v4, v58, v56
    v_add_u32_e32 v4, 8, v4
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v33
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v50, v0

    // Group 3 -> v51 (cols 12..15)
    v_add_u32_e32 v4, v58, v56
    v_add_u32_e32 v4, 12, v4
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v33
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v33
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v51, v0

    // Group 4 -> v52 (cols 16..19)
    v_add_u32_e32 v4, v58, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v32
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v52, v0

    // Group 5 -> v53 (cols 20..23)
    v_add_u32_e32 v4, v58, v56
    v_add_u32_e32 v4, 4, v4
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v32
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v53, v0

    // Group 6 -> v54 (cols 24..27)
    v_add_u32_e32 v4, v58, v56
    v_add_u32_e32 v4, 8, v4
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v32
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v54, v0

    // Group 7 -> v55 (cols 28..31)
    v_add_u32_e32 v4, v58, v56
    v_add_u32_e32 v4, 12, v4
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v32
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v32
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v55, v0

    // PV MFMA using TR8 V reads (K=64, tiles 0+1)
    // Triton-style base swizzle (bitop3:0x36 + XOR bases)
    v_lshlrev_b32_e32 v2, 6, v60         // v14 = tid << 6
    v_lshlrev_b32_e32 v3, 2, v60         // v141 = tid << 2
    v_and_b32_e32 v4, 48, v3
    v_and_or_b32 v2, v2, s25, v4         // v14 = (v14 & ~0xb80) | (v141 & 0xb80)
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v2, v2, v5, v6 bitop3:0x36  // base (relative)

    v_xor_b32_e32 v3, 0x20, v2
    v_xor_b32_e32 v4, 0x460, v2
    v_xor_b32_e32 v5, 0x1020, v2
    v_xor_b32_e32 v6, 0x1460, v2
    v_xor_b32_e32 v7, 0x60, v2
    v_xor_b32_e32 v8, 0x420, v2
    v_xor_b32_e32 v9, 0x1060, v2
    v_xor_b32_e32 v10, 0x1420, v2

    v_add_u32_e32 v2, s42, v2
    v_add_u32_e32 v3, s42, v3
    v_add_u32_e32 v4, s42, v4
    v_add_u32_e32 v5, s42, v5
    v_add_u32_e32 v6, s42, v6
    v_add_u32_e32 v7, s42, v7
    v_add_u32_e32 v8, s42, v8
    v_add_u32_e32 v9, s42, v9
    v_add_u32_e32 v10, s42, v10

    // Preserve TR8 base addresses (avoid clobber by reads)
    v_mov_b32_e32 v20, v2
    v_mov_b32_e32 v21, v3
    v_mov_b32_e32 v22, v4
    v_mov_b32_e32 v23, v5
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v7
    v_mov_b32_e32 v26, v8
    v_mov_b32_e32 v27, v9
    v_mov_b32_e32 v28, v10


    ds_read_b64_tr_b8 v[0:1], v20 offset:0
    ds_read_b64_tr_b8 v[2:3], v20 offset:1088
    ds_read_b64_tr_b8 v[4:5], v20 offset:4096
    ds_read_b64_tr_b8 v[6:7], v20 offset:5184
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[64:79], v[48:55], v[0:7], v[64:79]

    ds_read_b64_tr_b8 v[0:1], v21
    ds_read_b64_tr_b8 v[2:3], v22
    ds_read_b64_tr_b8 v[4:5], v23
    ds_read_b64_tr_b8 v[6:7], v24
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[80:95], v[48:55], v[0:7], v[80:95]

    ds_read_b64_tr_b8 v[0:1], v20 offset:64
    ds_read_b64_tr_b8 v[2:3], v20 offset:1024
    ds_read_b64_tr_b8 v[4:5], v20 offset:4160
    ds_read_b64_tr_b8 v[6:7], v20 offset:5120
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[96:111], v[48:55], v[0:7], v[96:111]

    ds_read_b64_tr_b8 v[0:1], v25
    ds_read_b64_tr_b8 v[2:3], v26
    ds_read_b64_tr_b8 v[4:5], v27
    ds_read_b64_tr_b8 v[6:7], v28
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x64_f8f6f4 v[112:127], v[48:55], v[0:7], v[112:127]

    // Prefetch next tile pair (if any)
    s_add_u32 s34, s30, 2
    s_cmp_ge_u32 s34, s20
    s_cbranch_scc1 PREFETCH_DONE

    s_add_u32 s38, s31, 8192
    s_add_u32 s39, s31, 12288

    v_mov_b32_e32 v2, 256
    v_cmp_lt_u32_e32 vcc, v60, v2
    s_and_saveexec_b64 s[22:23], vcc
    v_add_u32_e32 v13, s38, v61          // K global offset (next tile0)
    buffer_load_dwordx4 v[20:23], v13, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v14, s40, v61          // K_LDS0 + (tid&255)*16
    ds_write_b128 v14, v[20:23]

    v_add_u32_e32 v13, s39, v61          // K global offset (next tile1)
    buffer_load_dwordx4 v[20:23], v13, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v14, s41, v61          // K_LDS1 + (tid&255)*16
    ds_write_b128 v14, v[20:23]
    s_mov_b64 exec, s[22:23]

    // Swizzled V load for next pair into V_LDS0
    v_mov_b32_e32 v2, 256
    v_cmp_lt_u32_e32 vcc, v60, v2
    s_and_saveexec_b64 s[22:23], vcc
    v_lshrrev_b32_e32 v4, 3, v60          // row = tid >> 3 (0..31)
    v_and_b32_e32 v5, 7, v60              // col_block = tid & 7 (0..7)
    v_lshlrev_b32_e32 v4, 7, v4           // row * 128
    v_lshlrev_b32_e32 v5, 4, v5           // col_block * 16
    v_add_u32_e32 v2, v4, v5              // byte offset within V
    buffer_load_dwordx4 v[40:43], v2, s[16:19], s38 offen
    v_add_u32_e32 v3, 4096, v2            // row + 32
    buffer_load_dwordx4 v[44:47], v3, s[16:19], s38 offen

    // Triton-style LDS write swizzle (bitop3:0x78)
    s_movk_i32 s26, 0x70
    v_lshlrev_b32_e32 v4, 4, v60          // tid * 16 bytes
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x78
    v_add_u32_e32 v4, s42, v4
    s_waitcnt vmcnt(0)
    ds_write_b128 v4, v[40:43]
    ds_write_b128 v4, v[44:47] offset:4096
    s_mov_b64 exec, s[22:23]
    s_waitcnt lgkmcnt(0)

PREFETCH_DONE:
    // Advance to next K/V tile pair
    s_add_u32 s31, s31, 8192
    s_add_u32 s30, s30, 2
    s_branch K_LOOP

K_LOOP_END:
    // ------------------------------------------------------------------------
    // Store O (linear per-thread store, 64 floats = 256 bytes)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v50, 8, v60        // tid * 256 bytes
    v_add_u32_e32 v51, s29, v50

    buffer_store_dwordx4 v[64:67], v51, s[4:7], 0 offen
    v_add_u32_e32 v52, 16, v51
    buffer_store_dwordx4 v[68:71], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 32, v51
    buffer_store_dwordx4 v[72:75], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 48, v51
    buffer_store_dwordx4 v[76:79], v52, s[4:7], 0 offen

    v_add_u32_e32 v52, 64, v51
    buffer_store_dwordx4 v[80:83], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 80, v51
    buffer_store_dwordx4 v[84:87], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 96, v51
    buffer_store_dwordx4 v[88:91], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 112, v51
    buffer_store_dwordx4 v[92:95], v52, s[4:7], 0 offen

    v_add_u32_e32 v52, 128, v51
    buffer_store_dwordx4 v[96:99], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 144, v51
    buffer_store_dwordx4 v[100:103], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 160, v51
    buffer_store_dwordx4 v[104:107], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 176, v51
    buffer_store_dwordx4 v[108:111], v52, s[4:7], 0 offen

    v_add_u32_e32 v52, 192, v51
    buffer_store_dwordx4 v[112:115], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 208, v51
    buffer_store_dwordx4 v[116:119], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 224, v51
    buffer_store_dwordx4 v[120:123], v52, s[4:7], 0 offen
    v_add_u32_e32 v52, 240, v51
    buffer_store_dwordx4 v[124:127], v52, s[4:7], 0 offen

    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_scaffold
    .amdhsa_group_segment_fixed_size 50176
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 52
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 240
    .amdhsa_next_free_sgpr 44
    .amdhsa_accum_offset 220
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_scaffold
    .symbol: _fwd_fp8_scaffold.kd
    .kernarg_segment_size: 52
    .group_segment_fixed_size: 50176
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 44
    .vgpr_count: 240
    .max_flat_workgroup_size: 512
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V_ptr, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: num_k_tiles, .size: 4, .offset: 32, .value_kind: by_value}
      - {.name: stride_qh, .size: 4, .offset: 36, .value_kind: by_value}
      - {.name: stride_kh, .size: 4, .offset: 40, .value_kind: by_value}
      - {.name: stride_vh, .size: 4, .offset: 44, .value_kind: by_value}
      - {.name: stride_oh, .size: 4, .offset: 48, .value_kind: by_value}
...
.end_amdgpu_metadata

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set Q_LDS, 0                 // 128x128 (pitch-132)
.set K_LDS, 16896             // 32x128 (row-major)
.set LDS_SIZE, 24576          // Q + K (>= 20992)

.text
.globl _fwd_fp8_p_pack_dump
.p2align 8
.type _fwd_fp8_p_pack_dump,@function

_fwd_fp8_p_pack_dump:
    s_mov_b64 exec, -1

    // ------------------------------------------------------------------------
    // Load kernel arguments
    // ------------------------------------------------------------------------
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr (uint32)
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr (fp8)
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr (fp8)
    s_waitcnt lgkmcnt(0)
    s_barrier

    // Buffer descriptors (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000

    // ------------------------------------------------------------------------
    // Thread indexing (256 threads)
    // ------------------------------------------------------------------------
    v_mov_b32_e32 v60, v0                // tid (0-255)
    v_lshrrev_b32_e32 v9, 6, v60         // wave_id = tid / 64 (0-3)
    v_and_b32_e32 v10, 63, v60           // lane_id = tid % 64

    // ------------------------------------------------------------------------
    // Load Q tile to LDS (Triton swizzle)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v1, 6, v60         // tid * 64 bytes
    v_mov_b32_e32 v2, v1                 // Q global offset

    buffer_load_dwordx4 v[40:43], v2, s[8:11], 0 offen
    v_add_u32_e32 v3, 16, v2
    buffer_load_dwordx4 v[44:47], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 32, v2
    buffer_load_dwordx4 v[48:51], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 48, v2
    buffer_load_dwordx4 v[52:55], v3, s[8:11], 0 offen
    s_waitcnt vmcnt(0)

    // Q LDS address (pitch-132): row = tid>>1, col = (tid&1)*64
    v_and_b32_e32 v11, 0xFF, v60         // tid_in_tile (0-255)
    v_lshrrev_b32_e32 v11, 1, v11        // row
    v_lshlrev_b32_e32 v12, 7, v11        // row * 128
    v_lshlrev_b32_e32 v13, 2, v11        // row * 4
    v_add_u32_e32 v12, v12, v13          // row * 132
    v_and_b32_e32 v11, 1, v60
    v_lshlrev_b32_e32 v11, 6, v11        // col offset
    v_add_u32_e32 v20, v12, v11
    v_add_u32_e32 v20, Q_LDS, v20

    ds_write_b128 v20, v[40:43]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[44:47]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[48:51]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[52:55]

    // ------------------------------------------------------------------------
    // Load K tile to LDS (row-major, 256 threads)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v13, 4, v60        // tid * 16 bytes
    buffer_load_dwordx4 v[20:23], v13, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v38, K_LDS, v13
    ds_write_b128 v38, v[20:23]

    s_waitcnt lgkmcnt(0)
    s_barrier

    // ------------------------------------------------------------------------
    // Compute MFMA LDS read addresses (pitch-132 Q, row-major K)
    // ------------------------------------------------------------------------
    v_and_b32_e32 v11, 15, v10           // lane & 15
    v_lshrrev_b32_e32 v12, 4, v10
    v_and_b32_e32 v12, 1, v12
    v_lshlrev_b32_e32 v12, 4, v12
    v_add_u32_e32 v13, v11, v12          // mfma_row (0..31)

    v_lshlrev_b32_e32 v14, 7, v13        // row * 128
    v_lshlrev_b32_e32 v15, 2, v13        // row * 4
    v_add_u32_e32 v14, v14, v15          // Q row offset (row * 132)
    v_lshlrev_b32_e32 v18, 7, v13        // K row offset (row * 128)

    v_cmp_ge_u32_e64 vcc, v10, 32
    v_cndmask_b32_e64 v11, 0, 16, vcc    // k_off1
    v_cndmask_b32_e64 v12, 32, 48, vcc   // k_off2

    v_lshrrev_b32_e32 v9, 6, v60         // wave_id = tid / 64
    v_and_b32_e32 v15, 3, v9             // wave_in_tile
    v_lshlrev_b32_e32 v17, 12, v15
    v_lshlrev_b32_e32 v19, 7, v15
    v_add_u32_e32 v17, v17, v19          // wave_in_tile * 4224

    v_add_u32_e32 v58, v17, v14          // Q base + row
    v_add_u32_e32 v24, v58, v11          // Q addr1
    v_add_u32_e32 v25, v58, v12          // Q addr2

    v_add_u32_e32 v26, K_LDS, v18        // K base + row
    v_add_u32_e32 v27, v26, v11          // K addr1
    v_add_u32_e32 v28, v26, v12          // K addr2

    // ------------------------------------------------------------------------
    // QK MFMA (K=64 x 2)
    // ------------------------------------------------------------------------
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

    // ------------------------------------------------------------------------
    // Pack P0 (tile 0) to FP8 (v48-v51); tile1 zeroed (v52-v55)
    // ------------------------------------------------------------------------
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

    // Triton-style lane mix for P -> B operand layout (for mapping)
    v_lshlrev_b32_e32 v237, 2, v10
    v_xor_b32_e32 v237, 0x80, v237
    v_and_b32_e32 v238, 32, v10
    v_cmp_eq_u32_e32 vcc, 0, v238

    v_mov_b32_e32 v220, v48
    v_mov_b32_e32 v221, v49
    v_mov_b32_e32 v222, v50
    v_mov_b32_e32 v223, v51
    v_mov_b32_e32 v224, v52
    v_mov_b32_e32 v225, v53
    v_mov_b32_e32 v226, v54
    v_mov_b32_e32 v227, v55

    v_cndmask_b32_e32 v228, v220, v222, vcc
    ds_bpermute_b32 v228, v237, v228
    v_cndmask_b32_e32 v229, v221, v223, vcc
    ds_bpermute_b32 v230, v237, v229
    v_cndmask_b32_e32 v229, v224, v226, vcc
    ds_bpermute_b32 v233, v237, v229
    v_cndmask_b32_e32 v229, v225, v227, vcc
    s_waitcnt lgkmcnt(2)
    v_cndmask_b32_e32 v220, v228, v220, vcc
    v_cndmask_b32_e32 v228, v222, v228, vcc
    ds_bpermute_b32 v235, v237, v229

    v_cndmask_b32_e32 v229, v230, v221, vcc
    v_cndmask_b32_e32 v230, v223, v230, vcc
    v_cndmask_b32_e32 v231, v233, v224, vcc
    v_cndmask_b32_e32 v232, v226, v233, vcc
    v_cndmask_b32_e32 v233, v235, v225, vcc
    v_cndmask_b32_e32 v234, v227, v235, vcc

    v_mov_b32_e32 v48, v220
    v_mov_b32_e32 v49, v228
    v_mov_b32_e32 v50, v229
    v_mov_b32_e32 v51, v230
    v_mov_b32_e32 v52, v231
    v_mov_b32_e32 v53, v232
    v_mov_b32_e32 v54, v233
    v_mov_b32_e32 v55, v234

    s_branch STORE_PACKED

    // Preserve packed P sources for bpermute
    v_mov_b32_e32 v120, v48
    v_mov_b32_e32 v121, v49
    v_mov_b32_e32 v122, v50
    v_mov_b32_e32 v123, v51

    v_mov_b32_e32 v52, 0
    v_mov_b32_e32 v53, 0
    v_mov_b32_e32 v54, 0
    v_mov_b32_e32 v55, 0

    // ------------------------------------------------------------------------
    // Build A operand from packed P (byte-level transpose)
    // ------------------------------------------------------------------------
    // row = lane & 31
    v_and_b32_e32 v57, 31, v10

    // row_reg = (row >> 2) & 3
    v_lshrrev_b32_e32 v62, 2, v57
    v_and_b32_e32 v62, 3, v62

    // byte_shift = (row & 3) * 8
    v_and_b32_e32 v59, 3, v57
    v_lshlrev_b32_e32 v63, 3, v59

    // lane_offset = lane (no swap) for mapping
    v_mov_b32_e32 v56, v10

    // Select source reg for tile0 (v48..v51) by row_reg
    v_mov_b32_e32 v32, v48
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v32, v32, v49, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v32, v32, v50, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v32, v32, v51, vcc

    // Select source reg for tile1 (v52..v55) by row_reg
    v_mov_b32_e32 v33, v52
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v33, v33, v53, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v33, v33, v54, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v33, v33, v55, vcc

    // Select tile by lane bit5: lanes 0-31 use tile0, 32-63 use tile1
    v_and_b32_e32 v140, 32, v10
    v_cmp_eq_u32_e32 vcc, 0, v140
    v_cndmask_b32_e32 v32, v33, v32, vcc


    // Group 0 -> v48 (cols 0..3) using per-row source selection
    v_mov_b32_e32 v4, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v48, v0

    // Group 1 -> v49 (cols 4..7) using per-row source selection
    v_add_u32_e32 v4, 4, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v49, v0

    // Group 2 -> v50 (cols 8..11) using per-row source selection
    v_add_u32_e32 v4, 8, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v50, v0

    // Group 3 -> v51 (cols 12..15) using per-row source selection
    v_add_u32_e32 v4, 12, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v51, v0

    // Group 4 -> v52 (cols 16..19) using per-row source selection
    v_add_u32_e32 v4, 16, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v52, v0

    // Group 5 -> v53 (cols 20..23) using per-row source selection
    v_add_u32_e32 v4, 20, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v53, v0

    // Group 6 -> v54 (cols 24..27) using per-row source selection
    v_add_u32_e32 v4, 24, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v54, v0

    // Group 7 -> v55 (cols 28..31) using per-row source selection
    v_add_u32_e32 v4, 28, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v120
    ds_bpermute_b32 v8, v4, v121
    ds_bpermute_b32 v12, v4, v122
    ds_bpermute_b32 v16, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v120
    ds_bpermute_b32 v9, v4, v121
    ds_bpermute_b32 v13, v4, v122
    ds_bpermute_b32 v17, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v120
    ds_bpermute_b32 v10, v4, v121
    ds_bpermute_b32 v14, v4, v122
    ds_bpermute_b32 v18, v4, v123
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v120
    ds_bpermute_b32 v11, v4, v121
    ds_bpermute_b32 v15, v4, v122
    ds_bpermute_b32 v19, v4, v123
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v0, v63, v0
    v_and_b32_e32 v0, 0xFF, v0
    v_lshrrev_b32_e32 v8, v63, v8
    v_and_b32_e32 v8, 0xFF, v8
    v_lshrrev_b32_e32 v12, v63, v12
    v_and_b32_e32 v12, 0xFF, v12
    v_lshrrev_b32_e32 v16, v63, v16
    v_and_b32_e32 v16, 0xFF, v16
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v0, v0, v8, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v0, v0, v12, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v0, v0, v16, vcc
    v_lshrrev_b32_e32 v1, v63, v1
    v_and_b32_e32 v1, 0xFF, v1
    v_lshrrev_b32_e32 v9, v63, v9
    v_and_b32_e32 v9, 0xFF, v9
    v_lshrrev_b32_e32 v13, v63, v13
    v_and_b32_e32 v13, 0xFF, v13
    v_lshrrev_b32_e32 v17, v63, v17
    v_and_b32_e32 v17, 0xFF, v17
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v1, v1, v9, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v1, v1, v13, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v1, v1, v17, vcc
    v_lshrrev_b32_e32 v2, v63, v2
    v_and_b32_e32 v2, 0xFF, v2
    v_lshrrev_b32_e32 v10, v63, v10
    v_and_b32_e32 v10, 0xFF, v10
    v_lshrrev_b32_e32 v14, v63, v14
    v_and_b32_e32 v14, 0xFF, v14
    v_lshrrev_b32_e32 v18, v63, v18
    v_and_b32_e32 v18, 0xFF, v18
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v2, v2, v10, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v2, v2, v14, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v2, v2, v18, vcc
    v_lshrrev_b32_e32 v3, v63, v3
    v_and_b32_e32 v3, 0xFF, v3
    v_lshrrev_b32_e32 v11, v63, v11
    v_and_b32_e32 v11, 0xFF, v11
    v_lshrrev_b32_e32 v15, v63, v15
    v_and_b32_e32 v15, 0xFF, v15
    v_lshrrev_b32_e32 v19, v63, v19
    v_and_b32_e32 v19, 0xFF, v19
    v_cmp_eq_u32_e32 vcc, 1, v62
    v_cndmask_b32_e32 v3, v3, v11, vcc
    v_cmp_eq_u32_e32 vcc, 2, v62
    v_cndmask_b32_e32 v3, v3, v15, vcc
    v_cmp_eq_u32_e32 vcc, 3, v62
    v_cndmask_b32_e32 v3, v3, v19, vcc
    v_lshlrev_b32_e32 v1, 8, v1
    v_lshlrev_b32_e32 v2, 16, v2
    v_lshlrev_b32_e32 v3, 24, v3
    v_or_b32_e32 v0, v0, v1
    v_or_b32_e32 v0, v0, v2
    v_or_b32_e32 v0, v0, v3
    v_mov_b32_e32 v55, v0

    // Triton-style lane mix for packed P (debug)
    v_lshlrev_b32_e32 v237, 2, v10
    v_xor_b32_e32 v237, 0x80, v237
    v_and_b32_e32 v238, 32, v10
    v_cmp_eq_u32_e32 vcc, 0, v238

    // Preserve original packed regs for post-mix cndmask stage
    v_mov_b32_e32 v200, v48
    v_mov_b32_e32 v201, v49
    v_mov_b32_e32 v202, v50
    v_mov_b32_e32 v203, v51
    v_mov_b32_e32 v204, v52
    v_mov_b32_e32 v205, v53
    v_mov_b32_e32 v206, v54
    v_mov_b32_e32 v207, v55

    v_mov_b32_e32 v220, v48
    v_mov_b32_e32 v221, v49
    v_mov_b32_e32 v222, v50
    v_mov_b32_e32 v223, v51
    v_mov_b32_e32 v224, v52
    v_mov_b32_e32 v225, v53
    v_mov_b32_e32 v226, v54
    v_mov_b32_e32 v227, v55

    v_cndmask_b32_e32 v228, v220, v222, vcc
    ds_bpermute_b32 v228, v237, v228
    v_cndmask_b32_e32 v229, v221, v223, vcc
    ds_bpermute_b32 v230, v237, v229
    v_cndmask_b32_e32 v229, v224, v226, vcc
    ds_bpermute_b32 v233, v237, v229
    v_cndmask_b32_e32 v229, v225, v227, vcc
    s_waitcnt lgkmcnt(2)
    v_cndmask_b32_e32 v220, v228, v220, vcc
    v_cndmask_b32_e32 v228, v222, v228, vcc
    ds_bpermute_b32 v235, v237, v229

    v_cndmask_b32_e32 v229, v230, v221, vcc
    v_cndmask_b32_e32 v230, v223, v230, vcc
    v_cndmask_b32_e32 v231, v233, v224, vcc
    v_cndmask_b32_e32 v232, v226, v233, vcc
    v_cndmask_b32_e32 v233, v235, v225, vcc
    v_cndmask_b32_e32 v234, v227, v235, vcc

    v_mov_b32_e32 v48, v220
    v_mov_b32_e32 v49, v228
    v_mov_b32_e32 v50, v229
    v_mov_b32_e32 v51, v230
    v_mov_b32_e32 v52, v231
    v_mov_b32_e32 v53, v232
    v_mov_b32_e32 v54, v233
    v_mov_b32_e32 v55, v234

    // Post-mix selection stage (align with Triton final cndmask)
    v_and_b32_e32 v238, 32, v10
    v_cmp_eq_u32_e32 vcc, 0, v238
    v_cndmask_b32_e32 v50, v51, v201, vcc  // v68 = cndmask(v69, v82)
    v_cndmask_b32_e32 v51, v203, v51, vcc  // v69 = cndmask(v84, v69)
    v_cndmask_b32_e32 v52, v54, v204, vcc  // v70 = cndmask(v72, v85)
    v_cndmask_b32_e32 v53, v206, v54, vcc  // v71 = cndmask(v71, v72)
    v_cndmask_b32_e32 v54, v202, v205, vcc // v72 = cndmask(v83, v86)
    v_cndmask_b32_e32 v55, v207, v202, vcc // v73 = cndmask(v73, v83)




STORE_PACKED:
    // ------------------------------------------------------------------------
    // Store packed A operand (8 dwords per thread)
    // ------------------------------------------------------------------------
    v_lshlrev_b32_e32 v24, 5, v60        // tid * 32 bytes
    buffer_store_dwordx4 v[48:51], v24, s[4:7], 0 offen
    v_add_u32_e32 v25, 16, v24
    buffer_store_dwordx4 v[52:55], v25, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_p_pack_dump
    .amdhsa_group_segment_fixed_size 24576
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 240
    .amdhsa_next_free_sgpr 30
    .amdhsa_accum_offset 240
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_p_pack_dump
    .symbol: _fwd_fp8_p_pack_dump.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 24576
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 30
    .vgpr_count: 220
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

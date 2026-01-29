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

    // ISA requirement for correct BF8/FP8 math:
    // SH_MEM_CONFIG.bit[8] must be set to 1 (CDNA4 ISA §7.3).
    //
    // The assembler does not provide a named HW_REG_SH_MEM_CONFIG on this target,
    // so we use the numeric HWREG id for MEM_BASES (id=15) as a proxy that maps
    // to SH_MEM_CONFIG/SH_MEM_BASES on CDNA-family hardware.
    s_getreg_b32 s36, hwreg(15, 0, 32)
    s_or_b32 s36, s36, 0x00000100
    s_setreg_b32 hwreg(15, 0, 32), s36

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
    s_load_dword s34, s[0:1], 52          // debug_flags
    // PV TR8 read knobs (v_read_dump-compatible subset)
    s_load_dword s37, s[0:1], 56          // v_read_cb
    s_load_dword s44, s[0:1], 60          // v_read_lane_add (was v_read_lane_xor in v_read_dump)
    s_load_dword s45, s[0:1], 64          // v_read_v3_xor
    s_load_dword s46, s[0:1], 68          // v_read_v3_add
    s_load_dword s47, s[0:1], 72          // v_read_v4_add
    s_load_dword s48, s[0:1], 76          // v_read_v2_add
    s_load_dword s49, s[0:1], 80          // v_read_base_add
    s_load_dword s50, s[0:1], 84          // v_read_base_xor
    s_load_dword s51, s[0:1], 88          // v_read_base_extra_add
    s_load_dword s52, s[0:1], 92          // v_read_s25_override (0 = keep current s25)
    s_waitcnt lgkmcnt(0)
    s_mov_b32 s35, s34                    // debug flags (separate from stride)

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
    // Debug: dump cached V base at entry
    s_and_b32 s36, s35, 0x00000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_VBASE_ENTRY_DEBUG
    v_mov_b32_e32 v185, 0
    v_mov_b32_e32 v180, s42
    v_mov_b32_e32 v181, 0
    v_mov_b32_e32 v182, 0
    v_mov_b32_e32 v183, 0
    buffer_store_dwordx4 v[180:183], v185, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_VBASE_ENTRY_DEBUG:
    // Debug: dump LDS base constants
    s_and_b32 s36, s35, 0x20000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_LDS_BASE_DEBUG
    v_mov_b32_e32 v180, 0
    v_mov_b32_e32 v181, s40
    v_mov_b32_e32 v182, s41
    v_mov_b32_e32 v183, s42
    buffer_store_dwordx4 v[180:183], v180, s[4:7], 0 offen
    v_mov_b32_e32 v184, s43
    v_mov_b32_e32 v185, 0
    v_mov_b32_e32 v186, 0
    v_mov_b32_e32 v187, 0
    v_add_u32_e32 v180, 16, v180
    buffer_store_dwordx4 v[184:187], v180, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_LDS_BASE_DEBUG:

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
    // Make sure all lanes execute the zeroing. Some earlier sections use
    // s_and_saveexec for tid<256 LDS writes; we must not inherit a masked EXEC.
    s_mov_b64 exec, -1
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
    // If debug_flags 0x00000004 is set, use perm_id from v_read_cb like v_read_dump:
    //   perm_id = (v_read_cb >> 2) & 7
    s_and_b32 s36, s35, 0x00000004
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_VLOAD_PERM_ID
    v_mov_b32_e32 v212, s37
    v_lshrrev_b32_e32 v212, 2, v212
    v_and_b32_e32 v212, 7, v212
    v_cmp_eq_u32_e32 vcc, 0, v212
    s_cbranch_vccnz PERM_DONE_VLOAD
    v_cmp_eq_u32_e32 vcc, 1, v212
    s_cbranch_vccnz PERM_1_VLOAD
    v_cmp_eq_u32_e32 vcc, 2, v212
    s_cbranch_vccnz PERM_2_VLOAD
    v_cmp_eq_u32_e32 vcc, 3, v212
    s_cbranch_vccnz PERM_3_VLOAD
    v_cmp_eq_u32_e32 vcc, 4, v212
    s_cbranch_vccnz PERM_4_VLOAD
    v_cmp_eq_u32_e32 vcc, 5, v212
    s_cbranch_vccnz PERM_5_VLOAD
    s_branch PERM_DONE_VLOAD

PERM_1_VLOAD:
    // (b4,b3,b2)->(b3,b2,b4)
    v_and_b32_e32 v208, 8, v4
    v_lshlrev_b32_e32 v208, 1, v208          // b3 -> b4
    v_and_b32_e32 v209, 4, v4
    v_lshlrev_b32_e32 v209, 1, v209          // b2 -> b3
    v_and_b32_e32 v210, 16, v4
    v_lshrrev_b32_e32 v210, 2, v210          // b4 -> b2
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
    s_branch PERM_DONE_VLOAD

PERM_2_VLOAD:
    // (b4,b3,b2)->(b2,b4,b3)
    v_and_b32_e32 v208, 4, v4
    v_lshlrev_b32_e32 v208, 2, v208          // b2 -> b4
    v_and_b32_e32 v209, 16, v4
    v_lshrrev_b32_e32 v209, 1, v209          // b4 -> b3
    v_and_b32_e32 v210, 8, v4
    v_lshrrev_b32_e32 v210, 1, v210          // b3 -> b2
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
    s_branch PERM_DONE_VLOAD

PERM_3_VLOAD:
    // (b4,b3,b2)->(b3,b4,b2)
    v_and_b32_e32 v208, 8, v4
    v_lshlrev_b32_e32 v208, 1, v208          // b3 -> b4
    v_and_b32_e32 v209, 16, v4
    v_lshrrev_b32_e32 v209, 1, v209          // b4 -> b3
    v_and_b32_e32 v210, 4, v4                // b2 -> b2
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
    s_branch PERM_DONE_VLOAD

PERM_4_VLOAD:
    // (b4,b3,b2)->(b2,b3,b4)
    v_and_b32_e32 v208, 4, v4
    v_lshlrev_b32_e32 v208, 2, v208          // b2 -> b4
    v_and_b32_e32 v209, 8, v4                // b3 -> b3
    v_and_b32_e32 v210, 16, v4
    v_lshrrev_b32_e32 v210, 2, v210          // b4 -> b2
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
    s_branch PERM_DONE_VLOAD

PERM_5_VLOAD:
    // (b4,b3,b2)->(b4,b2,b3)
    v_and_b32_e32 v208, 16, v4               // b4 -> b4
    v_and_b32_e32 v209, 4, v4
    v_lshlrev_b32_e32 v209, 1, v209          // b2 -> b3
    v_and_b32_e32 v210, 8, v4
    v_lshrrev_b32_e32 v210, 1, v210          // b3 -> b2
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
    s_branch PERM_DONE_VLOAD

SKIP_VLOAD_PERM_ID:
    // Debug: disable row permutation for V global load (identity mapping).
    // Flag: debug_flags 0x00000080
    s_and_b32 s36, s35, 0x00000080
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc0 PERM_DONE_VLOAD
    // permute row bits: (b4,b3,b2)->(b3,b2,b4)
    v_and_b32_e32 v208, 8, v4
    v_lshlrev_b32_e32 v208, 1, v208
    v_and_b32_e32 v209, 4, v4
    v_lshlrev_b32_e32 v209, 1, v209
    v_and_b32_e32 v210, 16, v4
    v_lshrrev_b32_e32 v210, 2, v210
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
PERM_DONE_VLOAD:
    v_and_b32_e32 v5, 7, v60              // col_block = tid & 7 (0..7)
    v_lshlrev_b32_e32 v4, 7, v4           // row_perm * 128
    v_lshlrev_b32_e32 v5, 4, v5           // col_block * 16
    v_add_u32_e32 v2, v4, v5              // byte offset within V
    buffer_load_dwordx4 v[40:43], v2, s[16:19], s31 offen
    v_add_u32_e32 v3, 4096, v2            // row + 32
    buffer_load_dwordx4 v[44:47], v3, s[16:19], s31 offen
    s_waitcnt vmcnt(0)

    // Debug (B-isolation): force V bytes to FP8(1.0) before LDS write.
    // This guarantees PV A operand is all-ones in LDS, independent of TR8 mapping.
    // Flag: debug_flags 0x00010000
    s_and_b32 s36, s35, 0x00010000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_FORCE_V_ONES
    v_mov_b32_e32 v40, 0x38383838
    v_mov_b32_e32 v41, 0x38383838
    v_mov_b32_e32 v42, 0x38383838
    v_mov_b32_e32 v43, 0x38383838
    v_mov_b32_e32 v44, 0x38383838
    v_mov_b32_e32 v45, 0x38383838
    v_mov_b32_e32 v46, 0x38383838
    v_mov_b32_e32 v47, 0x38383838
SKIP_FORCE_V_ONES:

    // LDS write swizzle for V preload
    // Default: Triton-style (bitop3:0x78, C=0x70)
    // Solver option (debug_flags 0x00400000): bitop3:0x7a, C=0x0
    // If debug_flags 0x00000004 is set, use write_mode = (v_read_cb >> 8) & 3 (like v_read_dump):
    //   0 -> 0x78,c=0x70 ; 2 -> 0x7a,c=0
    s_movk_i32 s26, 0x70
    v_lshlrev_b32_e32 v4, 4, v60          // tid * 16 bytes
    // Debug: optionally skip swizzle (identity write)
    s_and_b32 s36, s35, 0x00080000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_WRITE_IDENTITY
    s_branch V_WRITE_ADDR_READY
SKIP_V_WRITE_IDENTITY:
    // v_read_cb write_mode path (preferred when enabled)
    s_and_b32 s36, s35, 0x00000004
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 V_WRITE_MODE_DEBUGFLAGS
    s_lshr_b32 s36, s37, 8
    s_and_b32 s36, s36, 3
    s_cmp_eq_u32 s36, 2
    s_cbranch_scc0 V_SWIZZLE_78
    s_mov_b32 s26, 0
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x7a
    s_branch V_WRITE_ADDR_READY
V_WRITE_MODE_DEBUGFLAGS:
    // Select swizzle ttbl based on debug_flags
    s_and_b32 s36, s35, 0x00400000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 V_SWIZZLE_78
    s_mov_b32 s26, 0
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x7a
    s_branch V_WRITE_ADDR_READY
V_SWIZZLE_78:
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x78
V_WRITE_ADDR_READY:
    v_add_u32_e32 v4, 41984, v4
    // Debug: dump V data (v40..v47) before LDS write
    s_and_b32 s36, s35, 0x08000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_DATA_DEBUG
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[40:43], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[44:47], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_V_DATA_DEBUG:
    // Debug: dump v4 and s42 immediately after add
    s_and_b32 s36, s35, 0x00800000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_ADDR_IMM_DEBUG
    v_lshlrev_b32_e32 v185, 5, v60        // tid * 32 bytes
    v_mov_b32_e32 v180, v4
    v_mov_b32_e32 v181, s42
    v_mov_b32_e32 v182, 0
    v_mov_b32_e32 v183, 0
    buffer_store_dwordx4 v[180:183], v185, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_V_ADDR_IMM_DEBUG:
    s_waitcnt vmcnt(0)
    // If only one K tile, duplicate row r into rows r+32/64/96
    s_cmp_eq_u32 s20, 1
    s_cbranch_scc0 SKIP_V_DUP0
    v_mov_b32_e32 v44, v40
    v_mov_b32_e32 v45, v41
    v_mov_b32_e32 v46, v42
    v_mov_b32_e32 v47, v43
SKIP_V_DUP0:
    // Debug: write V using solver layout (TR8 coverage booster)
    // Flag: debug_flags 0x00008000
    s_and_b32 s36, s35, 0x00008000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_TR8_LAYOUT
    // Solver result (tools/tr8_multi_base_solver.py, extended read set, lanes=256, offset_count=10):
    //   base0 = tr8_base(tid, s25=0xb80) ^ 0x0
    //   base1 = (tid<<3) ^ (tid<<5) ^ 0x0
    //   offsets = (0,256,512,768,1024,1152,1280,1408,2048,2176)
    //
    // base0 = tr8_base(tid)
    v_lshlrev_b32_e32 v2, 6, v60
    v_lshlrev_b32_e32 v3, 2, v60
    v_and_b32_e32 v5, 48, v3
    v_and_b32_e32 v6, 3, v60
    v_lshlrev_b32_e32 v6, 4, v6
    v_or_b32_e32 v5, v5, v6
    v_and_b32_e32 v2, 0xb80, v2
    v_or_b32_e32 v2, v2, v5
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v188, v2, v5, v6 bitop3:0x36
    v_add_u32_e32 v188, 41984, v188
    // base1 = (tid<<3)^(tid<<5)
    v_lshlrev_b32_e32 v189, 3, v60
    v_lshlrev_b32_e32 v190, 5, v60
    v_xor_b32_e32 v189, v189, v190
    v_add_u32_e32 v189, 41984, v189

    // writes at base0
    ds_write_b128 v188, v[40:43] offset:0
    ds_write_b128 v188, v[44:47] offset:256
    ds_write_b128 v188, v[40:43] offset:512
    ds_write_b128 v188, v[44:47] offset:768
    ds_write_b128 v188, v[40:43] offset:1024
    ds_write_b128 v188, v[44:47] offset:1152
    ds_write_b128 v188, v[40:43] offset:1280
    ds_write_b128 v188, v[44:47] offset:1408
    ds_write_b128 v188, v[40:43] offset:2048
    ds_write_b128 v188, v[44:47] offset:2176

    // writes at base1
    ds_write_b128 v189, v[40:43] offset:0
    ds_write_b128 v189, v[44:47] offset:256
    ds_write_b128 v189, v[40:43] offset:512
    ds_write_b128 v189, v[44:47] offset:768
    ds_write_b128 v189, v[40:43] offset:1024
    ds_write_b128 v189, v[44:47] offset:1152
    ds_write_b128 v189, v[40:43] offset:1280
    ds_write_b128 v189, v[44:47] offset:1408
    ds_write_b128 v189, v[40:43] offset:2048
    ds_write_b128 v189, v[44:47] offset:2176
    s_branch SKIP_V_TR8_WRITES
SKIP_V_TR8_LAYOUT:
    ds_write_b128 v4, v[40:43]
    ds_write_b128 v4, v[44:47] offset:4096
    s_cmp_eq_u32 s20, 1
    s_cbranch_scc0 SKIP_V_DUP0_EX
    ds_write_b128 v4, v[40:43] offset:8192
    ds_write_b128 v4, v[40:43] offset:12288
SKIP_V_DUP0_EX:
SKIP_V_TR8_WRITES:
    s_mov_b64 exec, s[22:23]
    s_waitcnt lgkmcnt(0)
    s_waitcnt vmcnt(0)

K_LOOP:
    s_waitcnt vmcnt(0)
    s_barrier
    // MFMA and LDS reads are not predicated by EXEC (MFMA ignores EXEC),
    // but many setup steps (ds_read, packing) are. Ensure we're not inheriting
    // a masked EXEC from earlier tid<256 load/store sections.
    s_mov_b64 exec, -1

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

    // Debug: dump QK accumulators (v32..v47) after tile0 MFMA and exit.
    // Flag: debug_flags 0x00040000
    s_and_b32 s36, s35, 0x00040000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_QK_ACC
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[32:35], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[36:39], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[40:43], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[44:47], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_QK_ACC:

    // Pack P0 (tile 0) to FP8 (v48-v51) - Triton-style (writes both halves explicitly)
    v_mov_b32_e32 v48, 0
    v_mov_b32_e32 v49, 0
    v_mov_b32_e32 v50, 0
    v_mov_b32_e32 v51, 0
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

    // Debug: dump tile0 QK accumulators (v32..v47) and packed tile0 P regs (v48..v51) and exit.
    // Layout per thread: 20 dwords (80B) = [v32..v47, v48..v51]
    // Flag: debug_flags 0x00000100
    s_and_b32 s36, s35, 0x00000100
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_TILE0_QK_AND_PACK
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64
    v_lshlrev_b32_e32 v181, 4, v60        // tid * 16
    v_add_u32_e32 v180, v180, v181        // tid * 80
    buffer_store_dwordx4 v[32:35], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[36:39], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[40:43], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[44:47], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 64, v180
    buffer_store_dwordx4 v[48:51], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_TILE0_QK_AND_PACK:

    // Recompute K row offset (v18) from lane_id (v10) to avoid relying on long-lived v18.
    // mfma_row = (lane & 15) + 16 * ((lane >> 4) & 1)
    v_and_b32_e32 v2, 15, v10
    v_lshrrev_b32_e32 v3, 4, v10
    v_and_b32_e32 v3, 1, v3
    v_lshlrev_b32_e32 v3, 4, v3
    v_add_u32_e32 v2, v2, v3
    v_lshlrev_b32_e32 v18, 7, v2         // row * 128

    // Compute Q/K read addresses for tile 1 (K tile1)
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

    // Debug: dump QK accumulators (v32..v47) after tile1 MFMA and exit.
    // Flag: debug_flags 0x00001000
    s_and_b32 s36, s35, 0x00001000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_QK_ACC1
    // Layout (per thread, 80B):
    //   +0:  v26..v29 (K base+row, K addr1, K addr2, K tile0 base)
    //   +16: v32..v35
    //   +32: v36..v39
    //   +48: v40..v43
    //   +64: v44..v47
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64
    v_lshlrev_b32_e32 v181, 5, v60        // tid * 32
    v_add_u32_e32 v180, v180, v181        // tid * 96 (enough spacing)
    buffer_store_dwordx4 v[26:29], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[32:35], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[36:39], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[40:43], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 64, v180
    buffer_store_dwordx4 v[44:47], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_QK_ACC1:

    // Pack P1 (tile 1) to FP8 (v52-v55) - Triton-style
    v_mov_b32_e32 v52, 0
    v_mov_b32_e32 v53, 0
    v_mov_b32_e32 v54, 0
    v_mov_b32_e32 v55, 0
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

    // Debug: dump QK accumulators (v32..v47) and packed P regs (v48..v55) and exit.
    // This is used for rigorous random-input B-path validation in Python.
    // Layout per thread: 24 dwords (96B) = [v32..v47, v48..v55]
    // Flag: debug_flags 0x00000200
    s_and_b32 s36, s35, 0x00000200
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_QK_AND_PACKEDP
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64
    v_lshlrev_b32_e32 v181, 5, v60        // tid * 32
    v_add_u32_e32 v180, v180, v181        // tid * 96
    // v32..v47 (16 dwords)
    buffer_store_dwordx4 v[32:35], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[36:39], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[40:43], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[44:47], v181, s[4:7], 0 offen
    // v48..v55 (8 dwords)
    v_add_u32_e32 v181, 64, v180
    buffer_store_dwordx4 v[48:51], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 80, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_QK_AND_PACKEDP:

    // Debug: dump packed P regs (v48..v55) BEFORE mix and exit.
    // Flag: debug_flags 0x00000800
    s_and_b32 s36, s35, 0x00000800
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_PACKED_P
    v_lshlrev_b32_e32 v180, 5, v60        // tid * 32 bytes
    buffer_store_dwordx4 v[48:51], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_PACKED_P:

    // Triton-style lane mix for P -> B operand layout (verbatim mapping with temps)
    v_lshlrev_b32_e32 v237, 2, v10
    v_xor_b32_e32 v237, 0x80, v237
    v_and_b32_e32 v238, 32, v10
    v_cmp_eq_u32_e32 vcc, 0, v238

    // Map packed P into temp regs (t66..t73, t82..t86)
    v_mov_b32_e32 v200, v48   // t66
    v_mov_b32_e32 v208, v49   // t82
    v_mov_b32_e32 v209, v50   // t83
    v_mov_b32_e32 v210, v51   // t84
    v_mov_b32_e32 v211, v52   // t85
    v_mov_b32_e32 v212, v53   // t86
    v_mov_b32_e32 v205, v54   // t71
    v_mov_b32_e32 v207, v55   // t73

    // Mix stage 1 (Triton)
    v_cndmask_b32_e32 v201, v200, v209, vcc      // t67
    ds_bpermute_b32 v201, v237, v201
    v_cndmask_b32_e32 v202, v208, v210, vcc      // t68
    ds_bpermute_b32 v203, v237, v202             // t69
    v_cndmask_b32_e32 v202, v211, v205, vcc
    ds_bpermute_b32 v206, v237, v202             // t72
    v_cndmask_b32_e32 v202, v212, v207, vcc
    s_waitcnt lgkmcnt(2)
    v_cndmask_b32_e32 v200, v201, v200, vcc
    v_cndmask_b32_e32 v201, v209, v201, vcc
    ds_bpermute_b32 v209, v237, v202             // t83
    s_waitcnt lgkmcnt(0)

    // Mix stage 2 (Triton)
    v_cndmask_b32_e32 v202, v203, v208, vcc
    v_cndmask_b32_e32 v203, v210, v203, vcc
    v_cndmask_b32_e32 v204, v206, v211, vcc      // t70
    v_cndmask_b32_e32 v205, v205, v206, vcc      // t71
    v_cndmask_b32_e32 v206, v209, v212, vcc      // t72
    v_cndmask_b32_e32 v207, v207, v209, vcc      // t73

    // Copy mixed P back to v48..v55 (B operand)
    v_mov_b32_e32 v48, v200
    v_mov_b32_e32 v49, v201
    v_mov_b32_e32 v50, v202
    v_mov_b32_e32 v51, v203
    v_mov_b32_e32 v52, v204
    v_mov_b32_e32 v53, v205
    v_mov_b32_e32 v54, v206
    v_mov_b32_e32 v55, v207

    // Debug: dump B operand regs (post-mix v48..v55) and exit.
    // Flag: debug_flags 0x00000400
    s_and_b32 s36, s35, 0x00000400
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_B_POSTMIX
    // Ensure all ds_bpermute traffic is complete before dumping.
    s_waitcnt lgkmcnt(0)
    v_lshlrev_b32_e32 v180, 5, v60        // tid * 32 bytes
    buffer_store_dwordx4 v[48:51], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_B_POSTMIX:

    s_nop 0
    // Mixed P is ready for use as MFMA B operand.
    s_branch P_A_TRANSPOSE

P_A_TRANSPOSE:
    // Build A operand from packed P (byte-level transpose) [disabled]
    // Preserve packed P sources for bpermute (tile0/1)
    v_mov_b32_e32 v200, v48
    v_mov_b32_e32 v201, v49
    v_mov_b32_e32 v202, v50
    v_mov_b32_e32 v203, v51
    v_mov_b32_e32 v204, v52
    v_mov_b32_e32 v205, v53
    v_mov_b32_e32 v206, v54
    v_mov_b32_e32 v207, v55

    // Packed P sources (full 8 regs)
    v_mov_b32_e32 v100, v200
    v_mov_b32_e32 v101, v201
    v_mov_b32_e32 v102, v202
    v_mov_b32_e32 v103, v203
    v_mov_b32_e32 v104, v204
    v_mov_b32_e32 v105, v205
    v_mov_b32_e32 v106, v206
    v_mov_b32_e32 v107, v207

    // row = lane & 31
    v_and_b32_e32 v57, 31, v10

    // row_reg = (row >> 2) & 3
    v_lshrrev_b32_e32 v62, 2, v57
    v_and_b32_e32 v62, 3, v62

    // byte_shift = (row & 3) * 8
    v_and_b32_e32 v58, 3, v57
    v_lshlrev_b32_e32 v63, 3, v58

    // lane_offset = 0 (mixed pack mostly lane==k)
    v_mov_b32_e32 v56, 0

    // Group 0 -> v48 (cols 16..19) using per-row source selection
    v_add_u32_e32 v4, 16, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 1 -> v49 (cols 20..23) using per-row source selection
    v_add_u32_e32 v4, 20, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 2 -> v50 (cols 24..27) using per-row source selection
    v_add_u32_e32 v4, 24, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 3 -> v51 (cols 28..31) using per-row source selection
    v_add_u32_e32 v4, 28, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 4 -> v52 (cols 0..3) using per-row source selection
    v_mov_b32_e32 v4, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 5 -> v53 (cols 4..7) using per-row source selection
    v_add_u32_e32 v4, 4, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 6 -> v54 (cols 8..11) using per-row source selection
    v_add_u32_e32 v4, 8, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Group 7 -> v55 (cols 12..15) using per-row source selection
    v_add_u32_e32 v4, 12, v56
    v_lshlrev_b32_e32 v4, 2, v4
    ds_bpermute_b32 v0, v4, v100
    ds_bpermute_b32 v8, v4, v101
    ds_bpermute_b32 v12, v4, v102
    ds_bpermute_b32 v16, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v1, v4, v100
    ds_bpermute_b32 v9, v4, v101
    ds_bpermute_b32 v13, v4, v102
    ds_bpermute_b32 v17, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v2, v4, v100
    ds_bpermute_b32 v10, v4, v101
    ds_bpermute_b32 v14, v4, v102
    ds_bpermute_b32 v18, v4, v103
    v_add_u32_e32 v4, 4, v4
    ds_bpermute_b32 v3, v4, v100
    ds_bpermute_b32 v11, v4, v101
    ds_bpermute_b32 v15, v4, v102
    ds_bpermute_b32 v19, v4, v103
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

    // Row5 override (mixed-pack anomalies)
    v_cmp_eq_u32_e32 vcc, 5, v57
    v_mov_b32_e32 v160, 0
    v_mov_b32_e32 v161, 0
    v_mov_b32_e32 v162, 0
    v_mov_b32_e32 v163, 0
    v_mov_b32_e32 v164, 0
    v_mov_b32_e32 v165, 0
    v_mov_b32_e32 v166, 0
    v_mov_b32_e32 v167, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v160, v160, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v160, v160, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v160, v160, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v160, v160, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v161, v161, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v161, v161, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v161, v161, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v161, v161, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v162, v162, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v162, v162, v180
    v_mov_b32_e32 v4, 232
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v162, v162, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v162, v162, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v163, v163, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v163, v163, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v163, v163, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v163, v163, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v164, v164, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v164, v164, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v164, v164, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v164, v164, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v165, v165, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v165, v165, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v165, v165, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v165, v165, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v166, v166, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v166, v166, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v166, v166, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v166, v166, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v167, v167, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v167, v167, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v167, v167, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v167, v167, v180
    v_cndmask_b32_e32 v48, v48, v160, vcc
    v_cndmask_b32_e32 v49, v49, v161, vcc
    v_cndmask_b32_e32 v50, v50, v162, vcc
    v_cndmask_b32_e32 v51, v51, v163, vcc
    v_cndmask_b32_e32 v52, v52, v164, vcc
    v_cndmask_b32_e32 v53, v53, v165, vcc
    v_cndmask_b32_e32 v54, v54, v166, vcc
    v_cndmask_b32_e32 v55, v55, v167, vcc

    // Row6 override (mixed-pack anomalies)
    v_cmp_eq_u32_e32 vcc, 6, v57
    v_mov_b32_e32 v168, 0
    v_mov_b32_e32 v169, 0
    v_mov_b32_e32 v170, 0
    v_mov_b32_e32 v171, 0
    v_mov_b32_e32 v172, 0
    v_mov_b32_e32 v173, 0
    v_mov_b32_e32 v174, 0
    v_mov_b32_e32 v175, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v168, v168, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v168, v168, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v168, v168, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v168, v168, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v169, v169, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v169, v169, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v169, v169, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v169, v169, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v170, v170, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v170, v170, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v170, v170, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v170, v170, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v171, v171, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v171, v171, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v171, v171, v180
    v_mov_b32_e32 v4, 228
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v171, v171, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v172, v172, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v172, v172, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v172, v172, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v172, v172, v180
    v_mov_b32_e32 v4, 152
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v173, v173, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v173, v173, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v173, v173, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v173, v173, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v174, v174, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v174, v174, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v174, v174, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v174, v174, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v175, v175, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v175, v175, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v175, v175, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v175, v175, v180
    v_cndmask_b32_e32 v48, v48, v168, vcc
    v_cndmask_b32_e32 v49, v49, v169, vcc
    v_cndmask_b32_e32 v50, v50, v170, vcc
    v_cndmask_b32_e32 v51, v51, v171, vcc
    v_cndmask_b32_e32 v52, v52, v172, vcc
    v_cndmask_b32_e32 v53, v53, v173, vcc
    v_cndmask_b32_e32 v54, v54, v174, vcc
    v_cndmask_b32_e32 v55, v55, v175, vcc


    // Row5 override (mixed-pack mapping)
    v_cmp_eq_u32_e32 vcc, 5, v57
    v_mov_b32_e32 v216, 0
    v_mov_b32_e32 v217, 0
    v_mov_b32_e32 v218, 0
    v_mov_b32_e32 v219, 0
    v_mov_b32_e32 v220, 0
    v_mov_b32_e32 v221, 0
    v_mov_b32_e32 v222, 0
    v_mov_b32_e32 v223, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 232
    ds_bpermute_b32 v180, v4, v54
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v223, v223, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v223, v223, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v223, v223, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v223, v223, v180
    v_cndmask_b32_e32 v48, v48, v216, vcc
    v_cndmask_b32_e32 v49, v49, v217, vcc
    v_cndmask_b32_e32 v50, v50, v218, vcc
    v_cndmask_b32_e32 v51, v51, v219, vcc
    v_cndmask_b32_e32 v52, v52, v220, vcc
    v_cndmask_b32_e32 v53, v53, v221, vcc
    v_cndmask_b32_e32 v54, v54, v222, vcc
    v_cndmask_b32_e32 v55, v55, v223, vcc

    // Row6 override (mixed-pack mapping)
    v_cmp_eq_u32_e32 vcc, 6, v57
    v_mov_b32_e32 v224, 0
    v_mov_b32_e32 v225, 0
    v_mov_b32_e32 v226, 0
    v_mov_b32_e32 v227, 0
    v_mov_b32_e32 v228, 0
    v_mov_b32_e32 v229, 0
    v_mov_b32_e32 v230, 0
    v_mov_b32_e32 v231, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 228
    ds_bpermute_b32 v180, v4, v54
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 152
    ds_bpermute_b32 v180, v4, v54
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v231, v231, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v231, v231, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v231, v231, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v49
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v231, v231, v180
    v_cndmask_b32_e32 v48, v48, v224, vcc
    v_cndmask_b32_e32 v49, v49, v225, vcc
    v_cndmask_b32_e32 v50, v50, v226, vcc
    v_cndmask_b32_e32 v51, v51, v227, vcc
    v_cndmask_b32_e32 v52, v52, v228, vcc
    v_cndmask_b32_e32 v53, v53, v229, vcc
    v_cndmask_b32_e32 v54, v54, v230, vcc
    v_cndmask_b32_e32 v55, v55, v231, vcc

    // Row0 override (mixed-pack mapping, v200 sources)
    v_cmp_eq_u32_e32 vcc, 0, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 220
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 196
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 136
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 252
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 252
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 196
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 160
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 252
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 248
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 156
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 252
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    // Row5 override (mixed-pack mapping, v200 sources)
    v_cmp_eq_u32_e32 vcc, 5, v57
    v_mov_b32_e32 v216, 0
    v_mov_b32_e32 v217, 0
    v_mov_b32_e32 v218, 0
    v_mov_b32_e32 v219, 0
    v_mov_b32_e32 v220, 0
    v_mov_b32_e32 v221, 0
    v_mov_b32_e32 v222, 0
    v_mov_b32_e32 v223, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v216, v216, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v217, v217, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 232
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v218, v218, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v219, v219, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v220, v220, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v221, v221, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v222, v222, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v223, v223, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v223, v223, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v223, v223, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v223, v223, v180
    v_cndmask_b32_e32 v48, v48, v216, vcc
    v_cndmask_b32_e32 v49, v49, v217, vcc
    v_cndmask_b32_e32 v50, v50, v218, vcc
    v_cndmask_b32_e32 v51, v51, v219, vcc
    v_cndmask_b32_e32 v52, v52, v220, vcc
    v_cndmask_b32_e32 v53, v53, v221, vcc
    v_cndmask_b32_e32 v54, v54, v222, vcc
    v_cndmask_b32_e32 v55, v55, v223, vcc

    // Row6 override (mixed-pack mapping, v200 sources)
    v_cmp_eq_u32_e32 vcc, 6, v57
    v_mov_b32_e32 v224, 0
    v_mov_b32_e32 v225, 0
    v_mov_b32_e32 v226, 0
    v_mov_b32_e32 v227, 0
    v_mov_b32_e32 v228, 0
    v_mov_b32_e32 v229, 0
    v_mov_b32_e32 v230, 0
    v_mov_b32_e32 v231, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v224, v224, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v225, v225, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v226, v226, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 228
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v227, v227, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v228, v228, v180
    v_mov_b32_e32 v4, 152
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v229, v229, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v230, v230, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v231, v231, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v231, v231, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v231, v231, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v201
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v231, v231, v180
    v_cndmask_b32_e32 v48, v48, v224, vcc
    v_cndmask_b32_e32 v49, v49, v225, vcc
    v_cndmask_b32_e32 v50, v50, v226, vcc
    v_cndmask_b32_e32 v51, v51, v227, vcc
    v_cndmask_b32_e32 v52, v52, v228, vcc
    v_cndmask_b32_e32 v53, v53, v229, vcc
    v_cndmask_b32_e32 v54, v54, v230, vcc
    v_cndmask_b32_e32 v55, v55, v231, vcc

    // Override rows 0..7 in A regs using p_pack_mapping_tile0.csv
    v_cmp_eq_u32_e32 vcc, 0, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 220
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 196
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 188
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 252
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 196
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 160
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 252
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 248
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 156
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 248
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 1, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 2, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 3, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v200
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 4, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 64
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 68
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 72
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 76
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 5, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 196
    ds_bpermute_b32 v180, v4, v207
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 80
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 84
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 88
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 92
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 6, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 152
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 0, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 7, v57
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v4, 96
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 100
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 104
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 108
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v4, 112
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 116
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 120
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 124
    ds_bpermute_b32 v180, v4, v204
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v206
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v4, 0
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 4
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 8
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 12
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v4, 16
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 20
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 24
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 28
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v4, 32
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 36
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 40
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 44
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v4, 48
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 52
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 56
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v4, 60
    ds_bpermute_b32 v180, v4, v205
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    // Debug: dump B operand regs (packed/mixed P) and exit.
    // Flag: debug_flags 0x00020000
    s_and_b32 s36, s35, 0x00020000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_B
    v_lshlrev_b32_e32 v180, 5, v60        // tid * 32 bytes
    buffer_store_dwordx4 v[48:51], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_B:

PV_MFMA_START:
    // PV MFMA using TR8 V reads (K=64, tiles 0+1)
    // PV MFMA expects B operand in v0..v7, but the base-address synthesis below clobbers v2..v10.
    // Preserve mixed/packed P (currently in v48..v55) into a stable temp range (v232..v239),
    // and restore into v0..v7 immediately before the PV MFMA.
    v_mov_b32_e32 v232, v48
    v_mov_b32_e32 v233, v49
    v_mov_b32_e32 v234, v50
    v_mov_b32_e32 v235, v51
    v_mov_b32_e32 v236, v52
    v_mov_b32_e32 v237, v53
    v_mov_b32_e32 v238, v54
    v_mov_b32_e32 v239, v55

    // TR8 base for PV A operand reads.
    // Default: historical "+1 lane" tweak.
    // Option (debug_flags 0x00000040): oracle-guided candidate
    //   - no +1 lane
    //   - add +16 to v4 term
    //   - s25 = 0xff0
    //
    // Option (debug_flags 0x00000004): v_read_dump-style PV base using v_read_dump-compatible knobs.
    // This is the preferred path for solver/bruteforce.
    s_and_b32 s36, s35, 0x00000004
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_VREADCB_BASE

    // Optional s25 override (0 = keep current s25)
    s_cmp_eq_u32 s52, 0
    s_cbranch_scc1 SKIP_S25_OVERRIDE_SCAFFOLD
    s_mov_b32 s25, s52
SKIP_S25_OVERRIDE_SCAFFOLD:

    // v61 = v60 + v_read_lane_add (s44)
    v_mov_b32_e32 v61, v60
    v_mov_b32_e32 v180, s44
    v_add_u32_e32 v61, v180, v61

    // v2/v3 terms from v61 (like v_read_dump)
    v_lshlrev_b32_e32 v2, 6, v61
    v_lshlrev_b32_e32 v3, 2, v61

    // v3 xor/add knobs
    v_mov_b32_e32 v180, s45
    v_xor_b32_e32 v3, v180, v3
    v_mov_b32_e32 v180, s46
    v_add_u32_e32 v3, v180, v3

    // colblk override from v_read_cb low bits: +((v_read_cb & 3) << 4)
    v_mov_b32_e32 v180, s37
    v_and_b32_e32 v180, 3, v180
    v_lshlrev_b32_e32 v180, 4, v180
    v_add_u32_e32 v3, v180, v3

    // v4 = (v3 & 48) + v_read_v4_add (s47)
    v_and_b32_e32 v4, 48, v3
    v_mov_b32_e32 v180, s47
    v_add_u32_e32 v4, v180, v4

    // v2 = (v2 & s25) | v4
    v_and_or_b32 v2, v2, s25, v4

    // v2 += v_read_v2_add (s48)
    v_mov_b32_e32 v180, s48
    v_add_u32_e32 v2, v180, v2

    // final tr8 base via bitop3:0x36 (same as v_read_dump)
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v2, v2, v5, v6 bitop3:0x36

    // fixed XOR sequence and add V_LDS0
    v_xor_b32_e32 v3, 0x20, v2
    v_xor_b32_e32 v4, 0x460, v2
    v_xor_b32_e32 v5, 0x1020, v2
    v_xor_b32_e32 v6, 0x1460, v2
    v_xor_b32_e32 v7, 0x60, v2
    v_xor_b32_e32 v8, 0x420, v2
    v_xor_b32_e32 v9, 0x1060, v2
    v_xor_b32_e32 v10, 0x1420, v2

    v_add_u32_e32 v2, 41984, v2
    v_add_u32_e32 v3, 41984, v3
    v_add_u32_e32 v4, 41984, v4
    v_add_u32_e32 v5, 41984, v5
    v_add_u32_e32 v6, 41984, v6
    v_add_u32_e32 v7, 41984, v7
    v_add_u32_e32 v8, 41984, v8
    v_add_u32_e32 v9, 41984, v9
    v_add_u32_e32 v10, 41984, v10

    // base_add and base_xor (v_read_dump order)
    v_mov_b32_e32 v180, s49
    v_add_u32_e32 v2, v180, v2
    v_add_u32_e32 v3, v180, v3
    v_add_u32_e32 v4, v180, v4
    v_add_u32_e32 v5, v180, v5
    v_add_u32_e32 v6, v180, v6
    v_add_u32_e32 v7, v180, v7
    v_add_u32_e32 v8, v180, v8
    v_add_u32_e32 v9, v180, v9
    v_add_u32_e32 v10, v180, v10
    v_mov_b32_e32 v180, s50
    v_xor_b32_e32 v2, v180, v2
    v_xor_b32_e32 v3, v180, v3
    v_xor_b32_e32 v4, v180, v4
    v_xor_b32_e32 v5, v180, v5
    v_xor_b32_e32 v6, v180, v6
    v_xor_b32_e32 v7, v180, v7
    v_xor_b32_e32 v8, v180, v8
    v_xor_b32_e32 v9, v180, v9
    v_xor_b32_e32 v10, v180, v10

    // Preserve TR8 base addresses (+ base_extra_add)
    v_mov_b32_e32 v20, v2
    v_mov_b32_e32 v21, v3
    v_mov_b32_e32 v22, v4
    v_mov_b32_e32 v23, v5
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v7
    v_mov_b32_e32 v26, v8
    v_mov_b32_e32 v27, v9
    v_mov_b32_e32 v28, v10
    v_mov_b32_e32 v180, s51
    v_add_u32_e32 v20, v180, v20
    v_add_u32_e32 v21, v180, v21
    v_add_u32_e32 v22, v180, v22
    v_add_u32_e32 v23, v180, v23
    v_add_u32_e32 v24, v180, v24
    v_add_u32_e32 v25, v180, v25
    v_add_u32_e32 v26, v180, v26
    v_add_u32_e32 v27, v180, v27
    v_add_u32_e32 v28, v180, v28
    s_branch PV_TR8_BASE_READY
SKIP_VREADCB_BASE:
    s_and_b32 s36, s35, 0x00000040
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 TR8_BASE_DEFAULT
    // Candidate path
    s_movk_i32 s25, 0xff0
    v_mov_b32_e32 v61, v60
    v_lshlrev_b32_e32 v2, 6, v61
    v_lshlrev_b32_e32 v3, 2, v61
    v_and_b32_e32 v4, 48, v3
    v_and_b32_e32 v180, 3, v60
    // Debug: disable low2 injection ((tid&3)<<4) into v4.
    // Flag: debug_flags 0x00000010
    s_and_b32 s36, s35, 0x00000010
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 TR8_LOW2_INJECT_CAND
    v_mov_b32_e32 v180, 0
    s_branch TR8_LOW2_DONE_CAND
TR8_LOW2_INJECT_CAND:
    v_lshlrev_b32_e32 v180, 4, v180
TR8_LOW2_DONE_CAND:
    v_or_b32_e32 v4, v4, v180
    v_add_u32_e32 v4, 16, v4
    v_and_or_b32 v2, v2, s25, v4
    s_branch TR8_BASE_READY
TR8_BASE_DEFAULT:
    // add +1 to the base lane before forming (tid<<6, tid<<2) terms.
    v_add_u32_e32 v61, 1, v60
    v_lshlrev_b32_e32 v2, 6, v61
    v_lshlrev_b32_e32 v3, 2, v61
    v_and_b32_e32 v4, 48, v3
    v_and_b32_e32 v180, 3, v60
    // Debug: disable low2 injection ((tid&3)<<4) into v4.
    // Flag: debug_flags 0x00000010
    s_and_b32 s36, s35, 0x00000010
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 TR8_LOW2_INJECT_DEF
    v_mov_b32_e32 v180, 0
    s_branch TR8_LOW2_DONE_DEF
TR8_LOW2_INJECT_DEF:
    v_lshlrev_b32_e32 v180, 4, v180
TR8_LOW2_DONE_DEF:
    v_or_b32_e32 v4, v4, v180
    v_and_or_b32 v2, v2, s25, v4
TR8_BASE_READY:
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v2, v2, v5, v6 bitop3:0x36

    // Solver (per-lane-bit brute force): inject lane bit0 at shift3 (xor 0x8 for odd lanes).
    // Flag: debug_flags 0x00000020
    s_and_b32 s36, s35, 0x00000020
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_LANE0_INJECT
    v_and_b32_e32 v180, 1, v60
    v_lshlrev_b32_e32 v180, 3, v180
    v_xor_b32_e32 v2, v2, v180
SKIP_TR8_LANE0_INJECT:

    // Debug: optionally use V-write swizzle base for TR8 reads
    s_and_b32 s36, s35, 0x00100000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_WRITE_BASE
    v_lshlrev_b32_e32 v2, 4, v60
    v_bitop3_b32 v2, v2, v60, s26 bitop3:0x78
SKIP_TR8_WRITE_BASE:
    // If using write-base override, match the selected V-write swizzle (0x78 vs 0x7a).
    // Recompute here because v_bitop3 uses an immediate ttbl.
    // (Only active when debug_flags 0x00100000 was set above.)
    // NOTE: this block is safe when not taken (falls through).
    s_and_b32 s36, s35, 0x00100000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 TR8_WRITE_BASE_DONE
    // Select swizzle based on debug_flags 0x00400000
    s_and_b32 s36, s35, 0x00400000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 TR8_WRITE_BASE_78
    v_lshlrev_b32_e32 v2, 4, v60
    v_bitop3_b32 v2, v2, v60, 0 bitop3:0x7a
    s_branch TR8_WRITE_BASE_DONE
TR8_WRITE_BASE_78:
    v_lshlrev_b32_e32 v2, 4, v60
    v_bitop3_b32 v2, v2, v60, s26 bitop3:0x78
TR8_WRITE_BASE_DONE:

    v_xor_b32_e32 v3, 0x20, v2
    v_xor_b32_e32 v4, 0x460, v2
    v_xor_b32_e32 v5, 0x1020, v2
    v_xor_b32_e32 v6, 0x1460, v2
    v_xor_b32_e32 v7, 0x60, v2
    v_xor_b32_e32 v8, 0x420, v2
    v_xor_b32_e32 v9, 0x1060, v2
    v_xor_b32_e32 v10, 0x1420, v2

    // Debug: optionally clamp TR8 offsets to 4KB window
    s_and_b32 s36, s35, 0x04000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_OFFSET_CLAMP
    v_and_b32_e32 v2, 0xFFF, v2
    v_and_b32_e32 v3, 0xFFF, v3
    v_and_b32_e32 v4, 0xFFF, v4
    v_and_b32_e32 v5, 0xFFF, v5
    v_and_b32_e32 v6, 0xFFF, v6
    v_and_b32_e32 v7, 0xFFF, v7
    v_and_b32_e32 v8, 0xFFF, v8
    v_and_b32_e32 v9, 0xFFF, v9
    v_and_b32_e32 v10, 0xFFF, v10
SKIP_TR8_OFFSET_CLAMP:

    v_add_u32_e32 v2, 41984, v2
    v_add_u32_e32 v3, 41984, v3
    v_add_u32_e32 v4, 41984, v4
    v_add_u32_e32 v5, 41984, v5
    v_add_u32_e32 v6, 41984, v6
    v_add_u32_e32 v7, 41984, v7
    v_add_u32_e32 v8, 41984, v8
    v_add_u32_e32 v9, 41984, v9
    v_add_u32_e32 v10, 41984, v10

    // Preserve TR8 base addresses
    v_mov_b32_e32 v20, v2
    v_mov_b32_e32 v21, v3
    v_mov_b32_e32 v22, v4
    v_mov_b32_e32 v23, v5
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v7
    v_mov_b32_e32 v26, v8
    v_mov_b32_e32 v27, v9
    v_mov_b32_e32 v28, v10
PV_TR8_BASE_READY:
    // Debug: dump TR8 base addresses v20..v28 and tid
    s_and_b32 s36, s35, 0x02000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_ADDR_DEBUG
    v_lshlrev_b32_e32 v185, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[20:23], v185, s[4:7], 0 offen
    v_add_u32_e32 v186, 16, v185
    buffer_store_dwordx4 v[24:27], v186, s[4:7], 0 offen
    v_mov_b32_e32 v180, v28
    v_mov_b32_e32 v181, v60
    v_mov_b32_e32 v182, 0
    v_mov_b32_e32 v183, 0
    v_add_u32_e32 v186, 32, v185
    buffer_store_dwordx4 v[180:183], v186, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_TR8_ADDR_DEBUG:

    // Debug (solver knob): per-base tweak for v23/v24 only.
    // Motivation: our raw TR8 dump shows missing ks clustered around the v23/v24-based reads:
    //   ds_read_b64_tr_b8 v[212:213], v23 offset:1280
    //   ds_read_b64_tr_b8 v[214:215], v24 offset:1408
    //
    // Encoding:
    //   t23 = (v_read_cb >> 12) & 0xF, apply v23 ^= (t23 << 5)
    //   t24 = (v_read_cb >> 16) & 0xF, apply v24 ^= (t24 << 5)
    // Active only when debug_flags 0x00000004 (v_read_cb base path) is enabled.
    s_and_b32 s36, s35, 0x00000004
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V24_TWEAK
    v_mov_b32_e32 v180, s37
    v_lshrrev_b32_e32 v180, 12, v180
    v_and_b32_e32 v180, 0xF, v180
    v_lshlrev_b32_e32 v180, 5, v180
    v_xor_b32_e32 v23, v23, v180
    v_mov_b32_e32 v180, s37
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xF, v180
    v_lshlrev_b32_e32 v180, 5, v180
    v_xor_b32_e32 v24, v24, v180
SKIP_V24_TWEAK:

    // Debug: sanity-check identity V LDS content at a fixed row.
    // Flag: debug_flags 0x40000000
    // Reads LDS at: V_LDS0 + 57 * 128 bytes = 41984 + 7296 = 49280
    // If identity V write is correct for rowbyte V[k,*]=k, this should read 0x39393939 (byte 57).
    s_and_b32 s36, s35, 0x40000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_LDS_ROW57_CHECK
    s_mov_b64 exec, -1
    v_mov_b32_e32 v180, 49280
    ds_read_b32 v240, v180 offset:0
    s_waitcnt lgkmcnt(0)
    v_lshlrev_b32_e32 v181, 2, v60        // tid * 4 bytes
    buffer_store_dword v240, v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_V_LDS_ROW57_CHECK:

    // ISA requirement for MFMA transpose LDS loads:
    // EXEC must be all-ones prior to DS_READ_*_TR_* instructions.
    s_mov_b64 exec, -1
    // ISA requirement: LDS address must be aligned to the data size.
    // For DS_READ_B64_TR_B8, enforce 8-byte alignment.
    v_and_b32_e32 v20, 0xFFFFFFF8, v20
    v_and_b32_e32 v21, 0xFFFFFFF8, v21
    v_and_b32_e32 v22, 0xFFFFFFF8, v22
    v_and_b32_e32 v23, 0xFFFFFFF8, v23
    v_and_b32_e32 v24, 0xFFFFFFF8, v24
    v_and_b32_e32 v25, 0xFFFFFFF8, v25
    v_and_b32_e32 v26, 0xFFFFFFF8, v26
    v_and_b32_e32 v27, 0xFFFFFFF8, v27
    v_and_b32_e32 v28, 0xFFFFFFF8, v28

    // TR8 A reads into v200..v231 (4 sets)
    ds_read_b64_tr_b8 v[200:201], v20 offset:0
    ds_read_b64_tr_b8 v[202:203], v20 offset:256
    ds_read_b64_tr_b8 v[204:205], v20 offset:512
    ds_read_b64_tr_b8 v[206:207], v20 offset:768
    ds_read_b64_tr_b8 v[208:209], v21 offset:1024
    ds_read_b64_tr_b8 v[210:211], v22 offset:1152
    ds_read_b64_tr_b8 v[212:213], v23 offset:1280

    // v24-based read: use offset:0 (the historical offset:1408 can go out-of-range for K=64 tiles).
    ds_read_b64_tr_b8 v[214:215], v24 offset:0
    ds_read_b64_tr_b8 v[216:217], v20 offset:2048
    ds_read_b64_tr_b8 v[218:219], v20 offset:2176
    ds_read_b64_tr_b8 v[220:221], v20 offset:2304
    ds_read_b64_tr_b8 v[222:223], v20 offset:2432
    ds_read_b64_tr_b8 v[224:225], v25 offset:3072
    ds_read_b64_tr_b8 v[226:227], v26 offset:3200
    ds_read_b64_tr_b8 v[228:229], v27 offset:3328
    ds_read_b64_tr_b8 v[230:231], v28 offset:3456

    // Debug (solver harness): optional extra TR8 read from selectable base + delta.
    // Enable with debug_flags 0x00800000. delta is encoded in v_read_cb bits [27:20]:
    //   delta_bytes = ((v_read_cb >> 20) & 0xFF) << 7   (128-byte granularity; one FP8 row for D=128B)
    // base_sel is encoded in v_read_cb bits [31:28]:
    //   0..8 select v20..v28 as the base address
    // Destination: v[232:233]. If enabled, raw-dump path will also store v232..v235.
    s_and_b32 s36, s35, 0x00800000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_EXTRA_READ
    // Extract base_sel into v190.
    v_mov_b32_e32 v190, s37
    v_lshrrev_b32_e32 v190, 28, v190
    v_and_b32_e32 v190, 0xF, v190

    // Select base v191 = v20..v28 based on base_sel (default v20).
    v_mov_b32_e32 v191, v20
    v_cmp_eq_u32_e32 vcc, 1, v190
    v_cndmask_b32_e32 v191, v191, v21, vcc
    v_cmp_eq_u32_e32 vcc, 2, v190
    v_cndmask_b32_e32 v191, v191, v22, vcc
    v_cmp_eq_u32_e32 vcc, 3, v190
    v_cndmask_b32_e32 v191, v191, v23, vcc
    v_cmp_eq_u32_e32 vcc, 4, v190
    v_cndmask_b32_e32 v191, v191, v24, vcc
    v_cmp_eq_u32_e32 vcc, 5, v190
    v_cndmask_b32_e32 v191, v191, v25, vcc
    v_cmp_eq_u32_e32 vcc, 6, v190
    v_cndmask_b32_e32 v191, v191, v26, vcc
    v_cmp_eq_u32_e32 vcc, 7, v190
    v_cndmask_b32_e32 v191, v191, v27, vcc
    v_cmp_eq_u32_e32 vcc, 8, v190
    v_cndmask_b32_e32 v191, v191, v28, vcc

    v_mov_b32_e32 v180, s37
    v_lshrrev_b32_e32 v180, 20, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 7, v180
    v_add_u32_e32 v191, v191, v180
    v_and_b32_e32 v191, 0xFFFFFFF8, v191
    ds_read_b64_tr_b8 v[232:233], v191 offset:0
SKIP_TR8_EXTRA_READ:
    s_waitcnt lgkmcnt(0)

    // Debug: dump raw TR8 read regs v200..v231
    s_and_b32 s36, s35, 0x01000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_DEBUG
    v_lshlrev_b32_e32 v180, 7, v60        // tid * 128 bytes
    buffer_store_dwordx4 v[200:203], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[204:207], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[208:211], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[212:215], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 64, v180
    buffer_store_dwordx4 v[216:219], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 80, v180
    buffer_store_dwordx4 v[220:223], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 96, v180
    buffer_store_dwordx4 v[224:227], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 112, v180
    buffer_store_dwordx4 v[228:231], v181, s[4:7], 0 offen

    // If extra read is enabled, append 16 bytes at +128B:
    //   [v232, v233, v190 (addr used), v_read_cb]
    s_and_b32 s36, s35, 0x00800000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_TR8_EXTRA_DUMP
    v_mov_b32_e32 v234, v191
    v_mov_b32_e32 v235, s37
    v_add_u32_e32 v181, 128, v180
    buffer_store_dwordx4 v[232:235], v181, s[4:7], 0 offen
SKIP_TR8_EXTRA_DUMP:
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_TR8_DEBUG:

    // Directly pack selected V->A regs into v48..v55 from raw TR8 read regs v200..v231.
    // This mapping was solved deterministically with rowbyte V[k,*]=k across all 512 threads
    // for the FP8 32x32x64 dense layout (CDNA4 ISA 7.1.5.1):
    //   lanes 0..31: [0..15, 32..47]
    //   lanes 32..63: [16..31, 48..63]
    //
    // Packed byte positions p0..p31 map to raw byte indices r (within the 128B raw dump):
    //   p0..p3   -> r0..r3   (v200 byte0..3)
    //   p4..p7   -> r4..r7   (v201 byte0..3)
    //   p8..p11  -> r14,r15,r22,r23  (v203 b2,b3; v205 b2,b3)
    //   p12..p15 -> r30,r31,r38,r39  (v207 b2,b3; v209 b2,b3)
    //   p16..p19 -> r103,r104,r105,r106 (v225 b3; v226 b0..2)
    //   p20..p23 -> r107,r108,r109,r110 (v226 b3; v227 b0..2)
    //   p24..p27 -> r111,r56,r48,r49 (v227 b3; v214 b0; v212 b0..1)
    //   p28..p31 -> r50,r51,r52,r53 (v212 b2..3; v213 b0..1)
    //
    // Note: this bypasses the older 4-group pack+select logic entirely.
    v_mov_b32_e32 v48, v200
    v_mov_b32_e32 v49, v201

    v_mov_b32_e32 v50, 0
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v50, v50, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v50, v50, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v50, v50, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v50, v50, v180

    v_mov_b32_e32 v51, 0
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v51, v51, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v51, v51, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v51, v51, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v51, v51, v180

    v_mov_b32_e32 v52, 0
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v52, v52, v180
    v_lshrrev_b32_e32 v180, 0, v226
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v52, v52, v180
    v_lshrrev_b32_e32 v180, 8, v226
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v52, v52, v180
    v_lshrrev_b32_e32 v180, 16, v226
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v52, v52, v180

    v_mov_b32_e32 v53, 0
    v_lshrrev_b32_e32 v180, 24, v226
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v53, v53, v180
    v_lshrrev_b32_e32 v180, 0, v227
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v53, v53, v180
    v_lshrrev_b32_e32 v180, 8, v227
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v53, v53, v180
    v_lshrrev_b32_e32 v180, 16, v227
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v53, v53, v180

    v_mov_b32_e32 v54, 0
    v_lshrrev_b32_e32 v180, 24, v227
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v54, v54, v180
    v_lshrrev_b32_e32 v180, 0, v214
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v54, v54, v180
    v_lshrrev_b32_e32 v180, 0, v212
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v54, v54, v180
    v_lshrrev_b32_e32 v180, 8, v212
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v54, v54, v180

    v_mov_b32_e32 v55, 0
    v_lshrrev_b32_e32 v180, 16, v212
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v55, v55, v180
    v_lshrrev_b32_e32 v180, 24, v212
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v55, v55, v180
    v_lshrrev_b32_e32 v180, 0, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v55, v55, v180
    v_lshrrev_b32_e32 v180, 8, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v55, v55, v180

    // Skip the older group0..3 pack and selection, but keep the early V->A dump point alive.
    s_branch VA_PACK_READY
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 24, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 16, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 24, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 24, v224
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 0, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 8, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 16, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 16, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 0, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 8, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 16, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 0, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 8, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 16, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 24, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 24, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v7, v7, v180
    v_lshrrev_b32_e32 v180, 0, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v7, v7, v180
    v_lshrrev_b32_e32 v180, 8, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v7, v7, v180
    v_lshrrev_b32_e32 v180, 16, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v7, v7, v180

    // Build A regs for group1 (lane16)
    v_mov_b32_e32 v240, 0
    v_mov_b32_e32 v241, 0
    v_mov_b32_e32 v242, 0
    v_mov_b32_e32 v243, 0
    v_mov_b32_e32 v244, 0
    v_mov_b32_e32 v245, 0
    v_mov_b32_e32 v246, 0
    v_mov_b32_e32 v247, 0
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 24, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 16, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 24, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 24, v224
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 0, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 8, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 16, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 16, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 0, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 8, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 16, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 0, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 8, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 16, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 24, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 24, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v247, v247, v180
    v_lshrrev_b32_e32 v180, 0, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v247, v247, v180
    v_lshrrev_b32_e32 v180, 8, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v247, v247, v180
    v_lshrrev_b32_e32 v180, 16, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v247, v247, v180
    // Build A regs for group2 (lane32)
    v_mov_b32_e32 v248, 0
    v_mov_b32_e32 v249, 0
    v_mov_b32_e32 v250, 0
    v_mov_b32_e32 v251, 0
    v_mov_b32_e32 v252, 0
    v_mov_b32_e32 v253, 0
    v_mov_b32_e32 v254, 0
    v_mov_b32_e32 v255, 0
    v_lshrrev_b32_e32 v180, 24, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v248, v248, v180
    v_lshrrev_b32_e32 v180, 16, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v248, v248, v180
    v_lshrrev_b32_e32 v180, 24, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v248, v248, v180
    v_lshrrev_b32_e32 v180, 24, v224
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v248, v248, v180
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v249, v249, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v249, v249, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v249, v249, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v249, v249, v180
    v_lshrrev_b32_e32 v180, 0, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v250, v250, v180
    v_lshrrev_b32_e32 v180, 8, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v250, v250, v180
    v_lshrrev_b32_e32 v180, 16, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v250, v250, v180
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v250, v250, v180
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v251, v251, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v251, v251, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v251, v251, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v251, v251, v180
    v_lshrrev_b32_e32 v180, 16, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v252, v252, v180
    v_lshrrev_b32_e32 v180, 0, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v252, v252, v180
    v_lshrrev_b32_e32 v180, 8, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v252, v252, v180
    v_lshrrev_b32_e32 v180, 16, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v252, v252, v180
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v253, v253, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v253, v253, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v253, v253, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v253, v253, v180
    v_lshrrev_b32_e32 v180, 24, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v254, v254, v180
    v_lshrrev_b32_e32 v180, 0, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v254, v254, v180
    v_lshrrev_b32_e32 v180, 8, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v254, v254, v180
    v_lshrrev_b32_e32 v180, 16, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v254, v254, v180
    v_lshrrev_b32_e32 v180, 0, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v255, v255, v180
    v_lshrrev_b32_e32 v180, 8, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v255, v255, v180
    v_lshrrev_b32_e32 v180, 16, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v255, v255, v180
    v_lshrrev_b32_e32 v180, 24, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v255, v255, v180
    // Build A regs for group3 (lane48)
    v_mov_b32_e32 v232, 0
    v_mov_b32_e32 v233, 0
    v_mov_b32_e32 v234, 0
    v_mov_b32_e32 v235, 0
    v_mov_b32_e32 v236, 0
    v_mov_b32_e32 v237, 0
    v_mov_b32_e32 v238, 0
    v_mov_b32_e32 v239, 0
    v_lshrrev_b32_e32 v180, 24, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v232, v232, v180
    v_lshrrev_b32_e32 v180, 16, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v232, v232, v180
    v_lshrrev_b32_e32 v180, 24, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v232, v232, v180
    v_lshrrev_b32_e32 v180, 24, v224
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v232, v232, v180
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v233, v233, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v233, v233, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v233, v233, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v233, v233, v180
    v_lshrrev_b32_e32 v180, 0, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v234, v234, v180
    v_lshrrev_b32_e32 v180, 8, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v234, v234, v180
    v_lshrrev_b32_e32 v180, 16, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v234, v234, v180
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v234, v234, v180
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v235, v235, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v235, v235, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v235, v235, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v235, v235, v180
    v_lshrrev_b32_e32 v180, 16, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v236, v236, v180
    v_lshrrev_b32_e32 v180, 0, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v236, v236, v180
    v_lshrrev_b32_e32 v180, 8, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v236, v236, v180
    v_lshrrev_b32_e32 v180, 16, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v236, v236, v180
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v237, v237, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v237, v237, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v237, v237, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v237, v237, v180
    v_lshrrev_b32_e32 v180, 24, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v238, v238, v180
    v_lshrrev_b32_e32 v180, 0, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v238, v238, v180
    v_lshrrev_b32_e32 v180, 8, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v238, v238, v180
    v_lshrrev_b32_e32 v180, 16, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v238, v238, v180
    v_lshrrev_b32_e32 v180, 0, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v239, v239, v180
    v_lshrrev_b32_e32 v180, 8, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v239, v239, v180
    v_lshrrev_b32_e32 v180, 16, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v239, v239, v180
    v_lshrrev_b32_e32 v180, 24, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v239, v239, v180

    // Select mapping by lane >= 32 (match v_read_dump)
    // Select lane-group-specific packed V->A regs and place in v48..v55 for PV MFMA:
    // - lanes  0..15  -> group0: v0..v7
    // - lanes 16..31  -> group1: v240..v247
    // - lanes 32..47  -> group2: v248..v255
    // - lanes 48..63  -> group3: v232..v239
    v_mov_b32_e32 v48, v0
    v_mov_b32_e32 v49, v1
    v_mov_b32_e32 v50, v2
    v_mov_b32_e32 v51, v3
    v_mov_b32_e32 v52, v4
    v_mov_b32_e32 v53, v5
    v_mov_b32_e32 v54, v6
    v_mov_b32_e32 v55, v7

    v_mov_b32_e32 v182, 16
    v_cmp_ge_u32_e32 vcc, v10, v182
    v_cndmask_b32_e32 v48, v48, v240, vcc
    v_cndmask_b32_e32 v49, v49, v241, vcc
    v_cndmask_b32_e32 v50, v50, v242, vcc
    v_cndmask_b32_e32 v51, v51, v243, vcc
    v_cndmask_b32_e32 v52, v52, v244, vcc
    v_cndmask_b32_e32 v53, v53, v245, vcc
    v_cndmask_b32_e32 v54, v54, v246, vcc
    v_cndmask_b32_e32 v55, v55, v247, vcc

    v_mov_b32_e32 v182, 32
    v_cmp_ge_u32_e32 vcc, v10, v182
    v_cndmask_b32_e32 v48, v48, v248, vcc
    v_cndmask_b32_e32 v49, v49, v249, vcc
    v_cndmask_b32_e32 v50, v50, v250, vcc
    v_cndmask_b32_e32 v51, v51, v251, vcc
    v_cndmask_b32_e32 v52, v52, v252, vcc
    v_cndmask_b32_e32 v53, v53, v253, vcc
    v_cndmask_b32_e32 v54, v54, v254, vcc
    v_cndmask_b32_e32 v55, v55, v255, vcc

    v_mov_b32_e32 v182, 48
    v_cmp_ge_u32_e32 vcc, v10, v182
    v_cndmask_b32_e32 v48, v48, v232, vcc
    v_cndmask_b32_e32 v49, v49, v233, vcc
    v_cndmask_b32_e32 v50, v50, v234, vcc
    v_cndmask_b32_e32 v51, v51, v235, vcc
    v_cndmask_b32_e32 v52, v52, v236, vcc
    v_cndmask_b32_e32 v53, v53, v237, vcc
    v_cndmask_b32_e32 v54, v54, v238, vcc
    v_cndmask_b32_e32 v55, v55, v239, vcc

VA_PACK_READY:
    // Debug: dump selected packed V->A regs (v48..v55) immediately after V-pack selection.
    // This is the *correct* place to dump V->A: later parts of the kernel reuse v0..v7
    // and the selection above would otherwise pick up non-V data (e.g. addresses).
    // Flag: debug_flags 0x00200000
    s_and_b32 s36, s35, 0x00200000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_V_A_SELECTED_EARLY
    v_lshlrev_b32_e32 v180, 5, v60        // tid * 32 bytes
    buffer_store_dwordx4 v[48:51], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_V_A_SELECTED_EARLY:

    // Debug: dump all 4 packed V->A groups (and selected) then exit.
    // Layout per tid: 40 dwords (160B):
    //   0..7   : group0 v0..v7
    //   8..15  : group1 v240..v247
    //   16..23 : group2 v248..v255
    //   24..31 : group3 v232..v239
    //   32..39 : selected v48..v55
    // Flag: debug_flags 0x00800000
    s_and_b32 s36, s35, 0x00800000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_V_A_GROUPS
    // tid * 160 bytes = tid*128 + tid*32
    v_lshlrev_b32_e32 v180, 7, v60        // tid * 128 bytes
    v_lshlrev_b32_e32 v181, 5, v60        // tid * 32 bytes
    v_add_u32_e32 v180, v180, v181
    buffer_store_dwordx4 v[0:3], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[4:7], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[240:243], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[244:247], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 64, v180
    buffer_store_dwordx4 v[248:251], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 80, v180
    buffer_store_dwordx4 v[252:255], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 96, v180
    buffer_store_dwordx4 v[232:235], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 112, v180
    buffer_store_dwordx4 v[236:239], v181, s[4:7], 0 offen
    // selected
    v_add_u32_e32 v181, 128, v180
    buffer_store_dwordx4 v[48:51], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 144, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_V_A_GROUPS:

    // Debug: dump selected packed V->A regs (v48..v55) and exit.
    // Flag: debug_flags 0x02000000
    s_and_b32 s36, s35, 0x02000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_V_A_SELECTED
    v_lshlrev_b32_e32 v180, 5, v60        // tid * 32 bytes
    buffer_store_dwordx4 v[48:51], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_V_A_SELECTED:

    // Fix B pos8..11 (k24..27) when only one K tile (last chance)
    s_cmp_eq_u32 s20, 1
    s_cbranch_scc0 SKIP_BPOS8_FIX_LATE
    v_mov_b32_e32 v2, v209
SKIP_BPOS8_FIX_LATE:

    // Legacy debug path: remap PV A regs from packed P (NOT V). This is only for
    // experimentation and must be disabled for real PV correctness.
    // Flag: debug_flags 0x20000000
    s_and_b32 s36, s35, 0x20000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_LEGACY_P_A_REMAP

    // Override row id for A remap: (lane & 15) + 16 * ((lane >> 5) & 1)
    v_and_b32_e32 v182, 15, v10
    v_lshrrev_b32_e32 v183, 5, v10
    v_and_b32_e32 v183, 1, v183
    v_lshlrev_b32_e32 v183, 4, v183
    v_add_u32_e32 v182, v183, v182

    // Override A regs for rows 0..15 using packed P half-swap
    v_cmp_gt_u32_e32 vcc, 16, v182
    v_cndmask_b32_e32 v48, v48, v104, vcc
    v_cndmask_b32_e32 v49, v49, v105, vcc
    v_cndmask_b32_e32 v50, v50, v106, vcc
    v_cndmask_b32_e32 v51, v51, v107, vcc
    v_cndmask_b32_e32 v52, v52, v100, vcc
    v_cndmask_b32_e32 v53, v53, v101, vcc
    v_cndmask_b32_e32 v54, v54, v102, vcc
    v_cndmask_b32_e32 v55, v55, v103, vcc

    // Override rows 16..31 using mixed P mapping (p_pack_mapping_random.csv)
    v_cmp_eq_u32_e32 vcc, 16, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 17, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 18, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 19, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v100
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 20, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 21, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 22, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 23, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v101
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 24, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 25, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 26, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 27, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v102
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 28, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 29, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 30, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    v_cmp_eq_u32_e32 vcc, 31, v182
    v_mov_b32_e32 v208, 0
    v_mov_b32_e32 v209, 0
    v_mov_b32_e32 v210, 0
    v_mov_b32_e32 v211, 0
    v_mov_b32_e32 v212, 0
    v_mov_b32_e32 v213, 0
    v_mov_b32_e32 v214, 0
    v_mov_b32_e32 v215, 0
    v_mov_b32_e32 v181, 128
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 132
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 136
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 140
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v212, v212, v180
    v_mov_b32_e32 v181, 144
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 148
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 152
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 156
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v213, v213, v180
    v_mov_b32_e32 v181, 160
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 164
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 168
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 172
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v214, v214, v180
    v_mov_b32_e32 v181, 176
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 180
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 184
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 188
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v215, v215, v180
    v_mov_b32_e32 v181, 192
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 196
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 200
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 204
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v208, v208, v180
    v_mov_b32_e32 v181, 208
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 212
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 216
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 220
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v209, v209, v180
    v_mov_b32_e32 v181, 224
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 228
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 232
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 236
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v210, v210, v180
    v_mov_b32_e32 v181, 240
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 244
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 248
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v211, v211, v180
    v_mov_b32_e32 v181, 252
    ds_bpermute_b32 v180, v181, v103
    s_waitcnt lgkmcnt(0)
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v211, v211, v180
    v_cndmask_b32_e32 v48, v48, v208, vcc
    v_cndmask_b32_e32 v49, v49, v209, vcc
    v_cndmask_b32_e32 v50, v50, v210, vcc
    v_cndmask_b32_e32 v51, v51, v211, vcc
    v_cndmask_b32_e32 v52, v52, v212, vcc
    v_cndmask_b32_e32 v53, v53, v213, vcc
    v_cndmask_b32_e32 v54, v54, v214, vcc
    v_cndmask_b32_e32 v55, v55, v215, vcc

    // Debug-only: re-pack V into MFMA A regs before PV MFMA (and before dumps).
    // This is expensive and intended only to validate/debug PV correctness.
    // Flag: debug_flags 0x10000000
    s_and_b32 s36, s35, 0x10000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DEBUG_REPACK_V_BEFORE_MFMA

    // Sanity marker to prove flag is observed.
    v_mov_b32_e32 v48, 0x07060504
    v_mov_b32_e32 v49, 0x0b0a0908
    // (Implementation is placed at the end of the kernel and reused here via branch.)
    s_branch DEBUG_REPACK_V_BLOCK

SKIP_DEBUG_REPACK_V_BEFORE_MFMA:
    // Debug: dump B regs (v0..v7) and A regs (v48..v55) and exit
    s_and_b32 s36, s35, 0x80000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 NO_A_DEBUG
    // Debug helper: encode marker into first A dword (inline constants)
    v_mov_b32_e32 v48, 1
    v_mov_b32_e32 v49, 2
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[0:3], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[4:7], v181, s[4:7], 0 offen
    v_add_u32_e32 v3, 32, v180
    buffer_store_dword v48, v3, s[4:7], 0 offen
    buffer_store_dwordx4 v[48:51], v3, s[4:7], 0 offen
    v_add_u32_e32 v3, 48, v180
    buffer_store_dwordx4 v[52:55], v3, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm

// ---------------------------------------------------------------------------
// Debug-only helper block (unreachable unless branched to)
// ---------------------------------------------------------------------------
DEBUG_REPACK_V_BLOCK:
    // Recompute PV TR8 base (v_read_dump-style path) into v20..v28, re-read TR8,
    // and pack lane0 mapping into v48..v55.

    // Optional s25 override (0 = keep current s25)
    s_cmp_eq_u32 s52, 0
    s_cbranch_scc1 SKIP_S25_OVERRIDE_DBG_REPACK
    s_mov_b32 s25, s52
SKIP_S25_OVERRIDE_DBG_REPACK:

    // v61 = v60 + v_read_lane_add (s44)
    v_mov_b32_e32 v61, v60
    v_mov_b32_e32 v180, s44
    v_add_u32_e32 v61, v180, v61

    // v2/v3 terms from v61 (like v_read_dump)
    v_lshlrev_b32_e32 v2, 6, v61
    v_lshlrev_b32_e32 v3, 2, v61

    // v3 xor/add knobs
    v_mov_b32_e32 v180, s45
    v_xor_b32_e32 v3, v180, v3
    v_mov_b32_e32 v180, s46
    v_add_u32_e32 v3, v180, v3

    // colblk override from v_read_cb low bits: +((v_read_cb & 3) << 4)
    v_mov_b32_e32 v180, s37
    v_and_b32_e32 v180, 3, v180
    v_lshlrev_b32_e32 v180, 4, v180
    v_add_u32_e32 v3, v180, v3

    // v4 = (v3 & 48) + v_read_v4_add (s47)
    v_and_b32_e32 v4, 48, v3
    v_mov_b32_e32 v180, s47
    v_add_u32_e32 v4, v180, v4

    // v2 = (v2 & s25) | v4
    v_and_or_b32 v2, v2, s25, v4

    // v2 += v_read_v2_add (s48)
    v_mov_b32_e32 v180, s48
    v_add_u32_e32 v2, v180, v2

    // final tr8 base via bitop3:0x36 (same as v_read_dump)
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v2, v2, v5, v6 bitop3:0x36

    // fixed XOR sequence and add V_LDS0
    v_xor_b32_e32 v3, 0x20, v2
    v_xor_b32_e32 v4, 0x460, v2
    v_xor_b32_e32 v5, 0x1020, v2
    v_xor_b32_e32 v6, 0x1460, v2
    v_xor_b32_e32 v7, 0x60, v2
    v_xor_b32_e32 v8, 0x420, v2
    v_xor_b32_e32 v9, 0x1060, v2
    v_xor_b32_e32 v10, 0x1420, v2

    v_add_u32_e32 v2, 41984, v2
    v_add_u32_e32 v3, 41984, v3
    v_add_u32_e32 v4, 41984, v4
    v_add_u32_e32 v5, 41984, v5
    v_add_u32_e32 v6, 41984, v6
    v_add_u32_e32 v7, 41984, v7
    v_add_u32_e32 v8, 41984, v8
    v_add_u32_e32 v9, 41984, v9
    v_add_u32_e32 v10, 41984, v10

    // base_add and base_xor (v_read_dump order)
    v_mov_b32_e32 v180, s49
    v_add_u32_e32 v2, v180, v2
    v_add_u32_e32 v3, v180, v3
    v_add_u32_e32 v4, v180, v4
    v_add_u32_e32 v5, v180, v5
    v_add_u32_e32 v6, v180, v6
    v_add_u32_e32 v7, v180, v7
    v_add_u32_e32 v8, v180, v8
    v_add_u32_e32 v9, v180, v9
    v_add_u32_e32 v10, v180, v10
    v_mov_b32_e32 v180, s50
    v_xor_b32_e32 v2, v180, v2
    v_xor_b32_e32 v3, v180, v3
    v_xor_b32_e32 v4, v180, v4
    v_xor_b32_e32 v5, v180, v5
    v_xor_b32_e32 v6, v180, v6
    v_xor_b32_e32 v7, v180, v7
    v_xor_b32_e32 v8, v180, v8
    v_xor_b32_e32 v9, v180, v9
    v_xor_b32_e32 v10, v180, v10

    // Preserve TR8 base addresses (+ base_extra_add)
    v_mov_b32_e32 v20, v2
    v_mov_b32_e32 v21, v3
    v_mov_b32_e32 v22, v4
    v_mov_b32_e32 v23, v5
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v7
    v_mov_b32_e32 v26, v8
    v_mov_b32_e32 v27, v9
    v_mov_b32_e32 v28, v10
    v_mov_b32_e32 v180, s51
    v_add_u32_e32 v20, v180, v20
    v_add_u32_e32 v21, v180, v21
    v_add_u32_e32 v22, v180, v22
    v_add_u32_e32 v23, v180, v23
    v_add_u32_e32 v24, v180, v24
    v_add_u32_e32 v25, v180, v25
    v_add_u32_e32 v26, v180, v26
    v_add_u32_e32 v27, v180, v27
    v_add_u32_e32 v28, v180, v28

    // ISA requirements for transpose LDS reads
    s_mov_b64 exec, -1
    v_and_b32_e32 v20, 0xFFFFFFF8, v20
    v_and_b32_e32 v21, 0xFFFFFFF8, v21
    v_and_b32_e32 v22, 0xFFFFFFF8, v22
    v_and_b32_e32 v23, 0xFFFFFFF8, v23
    v_and_b32_e32 v24, 0xFFFFFFF8, v24
    v_and_b32_e32 v25, 0xFFFFFFF8, v25
    v_and_b32_e32 v26, 0xFFFFFFF8, v26
    v_and_b32_e32 v27, 0xFFFFFFF8, v27
    v_and_b32_e32 v28, 0xFFFFFFF8, v28

    // Re-read TR8 A operand sets into v200..v231
    ds_read_b64_tr_b8 v[200:201], v20 offset:0
    ds_read_b64_tr_b8 v[202:203], v20 offset:256
    ds_read_b64_tr_b8 v[204:205], v20 offset:512
    ds_read_b64_tr_b8 v[206:207], v20 offset:768
    ds_read_b64_tr_b8 v[208:209], v21 offset:1024
    ds_read_b64_tr_b8 v[210:211], v22 offset:1152
    ds_read_b64_tr_b8 v[212:213], v23 offset:1280
    ds_read_b64_tr_b8 v[214:215], v24 offset:1408
    ds_read_b64_tr_b8 v[216:217], v20 offset:2048
    ds_read_b64_tr_b8 v[218:219], v20 offset:2176
    ds_read_b64_tr_b8 v[220:221], v20 offset:2304
    ds_read_b64_tr_b8 v[222:223], v20 offset:2432
    ds_read_b64_tr_b8 v[224:225], v25 offset:3072
    ds_read_b64_tr_b8 v[226:227], v26 offset:3200
    ds_read_b64_tr_b8 v[228:229], v27 offset:3328
    ds_read_b64_tr_b8 v[230:231], v28 offset:3456
    s_waitcnt lgkmcnt(0)

    // Pack lane0 mapping into v48..v55 (same as v_read_dump lane0 pack)
    v_mov_b32_e32 v48, 0
    v_mov_b32_e32 v49, 0
    v_mov_b32_e32 v50, 0
    v_mov_b32_e32 v51, 0
    v_mov_b32_e32 v52, 0
    v_mov_b32_e32 v53, 0
    v_mov_b32_e32 v54, 0
    v_mov_b32_e32 v55, 0
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v48, v48, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v48, v48, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v48, v48, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v48, v48, v180

    // Debug marker: force a recognizable pattern in v49 when repack runs.
    v_mov_b32_e32 v49, 0x07060504

    // Return to mainline (dump/MFMA)
    s_branch SKIP_DEBUG_REPACK_V_BEFORE_MFMA

NO_A_DEBUG:
SKIP_LEGACY_P_A_REMAP:
    // Debug sanity: if 0x10000000 set, force A bytes to FP8(1.0) right before MFMA.
    s_and_b32 s36, s35, 0x10000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_FORCE_A_ONES_BEFORE_MFMA
    v_mov_b32_e32 v48, 0x38383838
    v_mov_b32_e32 v49, 0x38383838
    v_mov_b32_e32 v50, 0x38383838
    v_mov_b32_e32 v51, 0x38383838
    v_mov_b32_e32 v52, 0x38383838
    v_mov_b32_e32 v53, 0x38383838
    v_mov_b32_e32 v54, 0x38383838
    v_mov_b32_e32 v55, 0x38383838
SKIP_FORCE_A_ONES_BEFORE_MFMA:
    // Debug sanity: dump v48 and exit.
    // Flag: debug_flags bit1 (0x00000002)
    s_and_b32 s36, s35, 0x00000002
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_V48_BEFORE_PV_MFMA
    v_mov_b32_e32 v0, 0
    buffer_store_dword v48, v0, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_V48_BEFORE_PV_MFMA:
    // Debug sanity: dump v0 (B operand word0) and exit.
    // Flag: debug_flags bit3 (0x00000008)
    s_and_b32 s36, s35, 0x00000008
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_V0_BEFORE_PV_MFMA
    v_mov_b32_e32 v1, v0
    v_mov_b32_e32 v0, 0
    buffer_store_dword v1, v0, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_V0_BEFORE_PV_MFMA:
    // Restore PV B operand into v0..v7 right before MFMA/dumps.
    v_mov_b32_e32 v0, v232
    v_mov_b32_e32 v1, v233
    v_mov_b32_e32 v2, v234
    v_mov_b32_e32 v3, v235
    v_mov_b32_e32 v4, v236
    v_mov_b32_e32 v5, v237
    v_mov_b32_e32 v6, v238
    v_mov_b32_e32 v7, v239

    // Debug: dump PV operands (B=v0..v7, A=v48..v55) per tid and exit.
    // Flag: debug_flags 0x00000001
    s_and_b32 s36, s35, 0x00000001
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_PV_AB
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[0:3], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[4:7], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 32, v180
    buffer_store_dwordx4 v[48:51], v181, s[4:7], 0 offen
    v_add_u32_e32 v181, 48, v180
    buffer_store_dwordx4 v[52:55], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_PV_AB:
    v_mfma_f32_32x32x64_f8f6f4 v[64:79], v[48:55], v[0:7], v[64:79]

    // Debug: dump PV accumulators after first MFMA (v64..v67) and exit.
    // Flag: debug_flags 0x00004000
    s_and_b32 s36, s35, 0x00004000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_DUMP_PV_ACC0
    v_lshlrev_b32_e32 v180, 4, v60        // tid * 16 bytes
    buffer_store_dwordx4 v[64:67], v180, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_DUMP_PV_ACC0:
    // Pack B regs for col block 1 (blockk+kbyte mapping)
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v180, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v202
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v216
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v218
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v220
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v222
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v201
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v201
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v203
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v217
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v217
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v217
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v219
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v231
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v224
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v1, v1, v180
    v_mov_b32_e32 v180, v228
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v3, v3, v180
    v_mov_b32_e32 v180, v225
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v3, v3, v180
    // Debug: dump B regs after block1 pack
    s_and_b32 s36, s35, 0x40000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_B1_DEBUG
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[0:3], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[4:7], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_B1_DEBUG:
    v_mfma_f32_32x32x64_f8f6f4 v[80:95], v[48:55], v[0:7], v[80:95]
    // Pack B regs for col block 2 (blockk+kbyte mapping)
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v180, v229
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v210
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v210
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v204
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v204
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v206
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v206
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v6, v6, v180
    v_mov_b32_e32 v180, v210
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v214
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v203
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v205
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v205
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v207
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v221
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v1, v1, v180
    v_mov_b32_e32 v180, v223
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v1, v1, v180
    v_mov_b32_e32 v180, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v2, v2, v180
    v_mov_b32_e32 v180, v207
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v2, v2, v180
    v_mov_b32_e32 v180, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v2, v2, v180
    v_mov_b32_e32 v180, v225
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v3, v3, v180
    v_mov_b32_e32 v180, v225
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v3, v3, v180
    v_mov_b32_e32 v180, v229
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v3, v3, v180
    // Debug: dump B regs after block2 pack
    s_and_b32 s36, s35, 0x20000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_B2_DEBUG
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[0:3], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[4:7], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_B2_DEBUG:
    v_mfma_f32_32x32x64_f8f6f4 v[96:111], v[48:55], v[0:7], v[96:111]
    // Pack B regs for col block 3 (blockk+kbyte mapping)
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v180, v200
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v200
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v202
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v202
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v4, v4, v180
    v_mov_b32_e32 v180, v216
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v216
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v218
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v220
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v5, v5, v180
    v_mov_b32_e32 v180, v214
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v214
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v7, v7, v180
    v_mov_b32_e32 v180, v203
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v205
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v0, v0, v180
    v_mov_b32_e32 v180, v215
    v_lshrrev_b32_e32 v180, 8, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v1, v1, v180
    v_mov_b32_e32 v180, v215
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v1, v1, v180
    v_mov_b32_e32 v180, v223
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v1, v1, v180
    v_mov_b32_e32 v180, v207
    v_lshrrev_b32_e32 v180, 16, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v2, v2, v180
    v_mov_b32_e32 v180, v209
    v_lshrrev_b32_e32 v180, 24, v180
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v2, v2, v180
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
    // permute row bits: (b4,b3,b2)->(b3,b2,b4)
    v_and_b32_e32 v208, 8, v4
    v_lshlrev_b32_e32 v208, 1, v208
    v_and_b32_e32 v209, 4, v4
    v_lshlrev_b32_e32 v209, 1, v209
    v_and_b32_e32 v210, 16, v4
    v_lshrrev_b32_e32 v210, 2, v210
    v_and_b32_e32 v211, 3, v4
    v_or_b32_e32 v208, v208, v209
    v_or_b32_e32 v208, v208, v210
    v_or_b32_e32 v4, v208, v211
    v_lshlrev_b32_e32 v4, 7, v4           // row_perm * 128
    v_lshlrev_b32_e32 v5, 4, v5           // col_block * 16
    v_add_u32_e32 v2, v4, v5              // byte offset within V
    buffer_load_dwordx4 v[40:43], v2, s[16:19], s38 offen
    v_add_u32_e32 v3, 4096, v2            // row + 32
    buffer_load_dwordx4 v[44:47], v3, s[16:19], s38 offen
    s_waitcnt vmcnt(0)

    // Triton-style LDS write swizzle (bitop3:0x78)
    s_movk_i32 s26, 0x70
    v_lshlrev_b32_e32 v4, 4, v60          // tid * 16 bytes
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x78
    s_mov_b32 s42, V_LDS0
    // Debug: dump s42 base and v4 before add
    // Flag: debug_flags 0x00002000
    s_and_b32 s36, s35, 0x00002000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_VBASE_PREADD_DEBUG
    v_mov_b32_e32 v180, s42
    v_mov_b32_e32 v181, v4
    v_mov_b32_e32 v182, 0
    v_mov_b32_e32 v183, 0
    v_mov_b32_e32 v185, 0
    buffer_store_dwordx4 v[180:183], v185, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_VBASE_PREADD_DEBUG:
    v_mov_b32_e32 v188, v4
    v_add_u32_e32 v188, 41984, v188
    // Debug: dump v188 after add
    s_and_b32 s36, s35, 0x00800000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_VADDR_ADD_DEBUG
    v_mov_b32_e32 v180, v188
    v_mov_b32_e32 v181, 0
    v_mov_b32_e32 v182, 0
    v_mov_b32_e32 v183, 0
    v_mov_b32_e32 v185, 0
    buffer_store_dwordx4 v[180:183], v185, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_VADDR_ADD_DEBUG:
    s_waitcnt vmcnt(0)
    // If only one K tile, duplicate row r into rows r+32/64/96
    s_cmp_eq_u32 s20, 1
    s_cbranch_scc0 SKIP_V_DUP1
    v_mov_b32_e32 v44, v40
    v_mov_b32_e32 v45, v41
    v_mov_b32_e32 v46, v42
    v_mov_b32_e32 v47, v43
SKIP_V_DUP1:
    // Debug: dump V LDS write addr and data (v40..v47)
    s_and_b32 s36, s35, 0x10000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_WRITE_DEBUG
    v_mov_b32_e32 v185, 0
    v_mov_b32_e32 v180, 41984             // expected V_LDS0 base
    v_mov_b32_e32 v181, v188              // LDS addr used for store
    v_mov_b32_e32 v182, v2                // global byte offset
    v_mov_b32_e32 v183, 0
    buffer_store_dwordx4 v[180:183], v185, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_V_WRITE_DEBUG:
    // Debug: dump V data (v40..v47) before LDS write
    s_and_b32 s36, s35, 0x08000000
    s_cmp_eq_u32 s36, 0
    s_cbranch_scc1 SKIP_V_DATA2_DEBUG
    v_lshlrev_b32_e32 v180, 6, v60        // tid * 64 bytes
    buffer_store_dwordx4 v[40:43], v180, s[4:7], 0 offen
    v_add_u32_e32 v181, 16, v180
    buffer_store_dwordx4 v[44:47], v181, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm
SKIP_V_DATA2_DEBUG:
    ds_write_b128 v188, v[40:43]
    ds_write_b128 v188, v[44:47] offset:4096
    s_cmp_eq_u32 s20, 1
    s_cbranch_scc0 SKIP_V_DUP1_EX
    ds_write_b128 v188, v[40:43] offset:8192
    ds_write_b128 v188, v[40:43] offset:12288
SKIP_V_DUP1_EX:
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
    .amdhsa_kernarg_size 96
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 56
    .amdhsa_accum_offset 256
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_scaffold
    .symbol: _fwd_fp8_scaffold.kd
    .kernarg_segment_size: 96
    .group_segment_fixed_size: 50176
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 56
    .vgpr_count: 256
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
      - {.name: debug_flags, .size: 4, .offset: 52, .value_kind: by_value}
      - {.name: v_read_cb, .size: 4, .offset: 56, .value_kind: by_value}
      - {.name: v_read_lane_add, .size: 4, .offset: 60, .value_kind: by_value}
      - {.name: v_read_v3_xor, .size: 4, .offset: 64, .value_kind: by_value}
      - {.name: v_read_v3_add, .size: 4, .offset: 68, .value_kind: by_value}
      - {.name: v_read_v4_add, .size: 4, .offset: 72, .value_kind: by_value}
      - {.name: v_read_v2_add, .size: 4, .offset: 76, .value_kind: by_value}
      - {.name: v_read_base_add, .size: 4, .offset: 80, .value_kind: by_value}
      - {.name: v_read_base_xor, .size: 4, .offset: 84, .value_kind: by_value}
      - {.name: v_read_base_extra_add, .size: 4, .offset: 88, .value_kind: by_value}
      - {.name: v_read_s25_override, .size: 4, .offset: 92, .value_kind: by_value}
...
.end_amdgpu_metadata

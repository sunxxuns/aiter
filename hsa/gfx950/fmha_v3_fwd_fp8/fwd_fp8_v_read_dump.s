// Debug kernel: dump PV A-operand reads from V LDS
.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set V_LDS0, 0                // debug: base at 0
.set V_LDS1, 4096             // V_LDS0 + 4096

.text
.globl _fwd_fp8_v_read_dump
.p2align 8
.type _fwd_fp8_v_read_dump,@function

_fwd_fp8_v_read_dump:
    s_mov_b64 exec, -1

    // Args: out_ptr, V_ptr, stride_vh (bytes)
    s_load_dwordx2 s[4:5], s[0:1], 0
    s_load_dwordx2 s[8:9], s[0:1], 8
    s_load_dword s12, s[0:1], 16
    s_waitcnt lgkmcnt(0)

    // Buffer descriptors
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_movk_i32 s25, 0xb80

    // tid
    v_mov_b32_e32 v60, v0

    // Load V tiles to LDS (swizzled, like scaffold)
    v_mov_b32_e32 v30, 256
    v_cmp_lt_u32_e32 vcc, v60, v30
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

    buffer_load_dwordx4 v[40:43], v2, s[8:11], 0 offen
    v_add_u32_e32 v3, 4096, v2            // row + 32
    buffer_load_dwordx4 v[44:47], v3, s[8:11], 0 offen
    v_add_u32_e32 v4, 8192, v2            // row + 64
    buffer_load_dwordx4 v[48:51], v4, s[8:11], 0 offen
    v_add_u32_e32 v5, 12288, v2           // row + 96
    buffer_load_dwordx4 v[52:55], v5, s[8:11], 0 offen

    // Triton-style LDS write swizzle (bitop3:0x78)
    s_movk_i32 s26, 0x70
    v_lshlrev_b32_e32 v4, 4, v60          // tid * 16 bytes
    v_bitop3_b32 v4, v4, v60, s26 bitop3:0x78
    v_add_u32_e32 v4, V_LDS0, v4
    s_waitcnt vmcnt(0)
    ds_write_b128 v4, v[40:43]
    ds_write_b128 v4, v[44:47] offset:4096
    ds_write_b128 v4, v[48:51] offset:8192
    ds_write_b128 v4, v[52:55] offset:12288
    s_mov_b64 exec, s[22:23]
    s_waitcnt lgkmcnt(0)
    s_barrier

    // PV read base (scaffold-style)
    v_lshlrev_b32_e32 v2, 6, v60
    v_lshlrev_b32_e32 v3, 2, v60
    v_and_b32_e32 v4, 48, v3
    v_and_or_b32 v2, v2, s25, v4
    v_and_b32_e32 v5, 16, v60
    v_lshlrev_b32_e32 v6, 3, v60
    v_and_b32_e32 v6, 8, v6
    v_bitop3_b32 v2, v2, v5, v6 bitop3:0x36

    // no lane-bit injection

    v_xor_b32_e32 v3, 0x20, v2
    v_xor_b32_e32 v4, 0x460, v2
    v_xor_b32_e32 v5, 0x1020, v2
    v_xor_b32_e32 v6, 0x1460, v2
    v_xor_b32_e32 v7, 0x60, v2
    v_xor_b32_e32 v8, 0x420, v2
    v_xor_b32_e32 v9, 0x1060, v2
    v_xor_b32_e32 v10, 0x1420, v2

    v_add_u32_e32 v2, V_LDS0, v2
    v_add_u32_e32 v3, V_LDS0, v3
    v_add_u32_e32 v4, V_LDS0, v4
    v_add_u32_e32 v5, V_LDS0, v5
    v_add_u32_e32 v6, V_LDS0, v6
    v_add_u32_e32 v7, V_LDS0, v7
    v_add_u32_e32 v8, V_LDS0, v8
    v_add_u32_e32 v9, V_LDS0, v9
    v_add_u32_e32 v10, V_LDS0, v10

    // Preserve base addresses like scaffold
    v_mov_b32_e32 v20, v2
    v_mov_b32_e32 v21, v3
    v_mov_b32_e32 v22, v4
    v_mov_b32_e32 v23, v5
    v_mov_b32_e32 v24, v6
    v_mov_b32_e32 v25, v7
    v_mov_b32_e32 v26, v8
    v_mov_b32_e32 v27, v9
    v_mov_b32_e32 v28, v10

    // Read A operand sets (same pattern as PV)
    // Set 0: v20 offsets
    ds_read_b64_tr_b8 v[0:1], v20 offset:0
    ds_read_b64_tr_b8 v[2:3], v20 offset:256
    ds_read_b64_tr_b8 v[4:5], v20 offset:512
    ds_read_b64_tr_b8 v[6:7], v20 offset:768
    s_waitcnt lgkmcnt(0)

    // Store set0 to output (per-thread, 40 dwords = 160 bytes)
    v_lshlrev_b32_e32 v30, 7, v60        // tid * 128 bytes
    v_lshlrev_b32_e32 v31, 5, v60        // tid * 32 bytes
    v_add_u32_e32 v30, v30, v31          // tid * 160 bytes
    buffer_store_dwordx4 v[0:3], v30, s[4:7], 0 offen
    v_add_u32_e32 v31, 16, v30
    buffer_store_dwordx4 v[4:7], v31, s[4:7], 0 offen

    // Set 1: v21..v24
    ds_read_b64_tr_b8 v[0:1], v21 offset:1024
    ds_read_b64_tr_b8 v[2:3], v22 offset:1152
    ds_read_b64_tr_b8 v[4:5], v23 offset:1280
    ds_read_b64_tr_b8 v[6:7], v24 offset:1408
    s_waitcnt lgkmcnt(0)

    v_add_u32_e32 v30, 32, v30
    buffer_store_dwordx4 v[0:3], v30, s[4:7], 0 offen
    v_add_u32_e32 v31, 16, v30
    buffer_store_dwordx4 v[4:7], v31, s[4:7], 0 offen

    // Set 2: v20 offsets (second half)
    ds_read_b64_tr_b8 v[0:1], v20 offset:2048
    ds_read_b64_tr_b8 v[2:3], v20 offset:2176
    ds_read_b64_tr_b8 v[4:5], v20 offset:2304
    ds_read_b64_tr_b8 v[6:7], v20 offset:2432
    s_waitcnt lgkmcnt(0)

    v_add_u32_e32 v30, 32, v30
    buffer_store_dwordx4 v[0:3], v30, s[4:7], 0 offen
    v_add_u32_e32 v31, 16, v30
    buffer_store_dwordx4 v[4:7], v31, s[4:7], 0 offen

    // Set 3: v25..v28
    ds_read_b64_tr_b8 v[0:1], v25 offset:3072
    ds_read_b64_tr_b8 v[2:3], v26 offset:3200
    ds_read_b64_tr_b8 v[4:5], v27 offset:3328
    ds_read_b64_tr_b8 v[6:7], v28 offset:3456
    s_waitcnt lgkmcnt(0)

    v_add_u32_e32 v30, 32, v30
    buffer_store_dwordx4 v[0:3], v30, s[4:7], 0 offen
    v_add_u32_e32 v31, 16, v30
    buffer_store_dwordx4 v[4:7], v31, s[4:7], 0 offen

    // Re-read TR8 A operand sets into v200..v231 for packing
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

    // Build A regs (k0..31) for lane0
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v0, v0, v180
    v_lshrrev_b32_e32 v180, 16, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 0, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 8, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 16, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v1, v1, v180
    v_lshrrev_b32_e32 v180, 0, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 8, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 16, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 24, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v2, v2, v180
    v_lshrrev_b32_e32 v180, 24, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 0, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 8, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 16, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v3, v3, v180
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v4, v4, v180
    v_lshrrev_b32_e32 v180, 24, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 16, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 24, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 24, v224
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v5, v5, v180
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v6, v6, v180
    v_lshrrev_b32_e32 v180, 0, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v7, v7, v180
    v_lshrrev_b32_e32 v180, 8, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v7, v7, v180
    v_lshrrev_b32_e32 v180, 16, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v7, v7, v180
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v7, v7, v180

    // Build A regs (k0..31) for lane32
    v_mov_b32_e32 v240, 0
    v_mov_b32_e32 v241, 0
    v_mov_b32_e32 v242, 0
    v_mov_b32_e32 v243, 0
    v_mov_b32_e32 v244, 0
    v_mov_b32_e32 v245, 0
    v_mov_b32_e32 v246, 0
    v_mov_b32_e32 v247, 0
    v_lshrrev_b32_e32 v180, 16, v213
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 0, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 8, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 16, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v240, v240, v180
    v_lshrrev_b32_e32 v180, 0, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 8, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 16, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 24, v200
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v241, v241, v180
    v_lshrrev_b32_e32 v180, 24, v210
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 0, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 8, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 16, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v242, v242, v180
    v_lshrrev_b32_e32 v180, 0, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 8, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 16, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 24, v201
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v243, v243, v180
    v_lshrrev_b32_e32 v180, 24, v211
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 16, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 24, v215
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 24, v224
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v244, v244, v180
    v_lshrrev_b32_e32 v180, 16, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 24, v203
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 16, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 24, v205
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v245, v245, v180
    v_lshrrev_b32_e32 v180, 0, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 8, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 16, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 24, v225
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v246, v246, v180
    v_lshrrev_b32_e32 v180, 16, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_or_b32_e32 v247, v247, v180
    v_lshrrev_b32_e32 v180, 24, v207
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 8, v180
    v_or_b32_e32 v247, v247, v180
    v_lshrrev_b32_e32 v180, 16, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 16, v180
    v_or_b32_e32 v247, v247, v180
    v_lshrrev_b32_e32 v180, 24, v209
    v_and_b32_e32 v180, 0xFF, v180
    v_lshlrev_b32_e32 v180, 24, v180
    v_or_b32_e32 v247, v247, v180

    // Select mapping by lane group (lane >= 32 -> lane32 mapping)
    v_mov_b32_e32 v182, 32
    v_cmp_ge_u32_e32 vcc, v60, v182
    v_cndmask_b32_e32 v0, v0, v240, vcc
    v_cndmask_b32_e32 v1, v1, v241, vcc
    v_cndmask_b32_e32 v2, v2, v242, vcc
    v_cndmask_b32_e32 v3, v3, v243, vcc
    v_cndmask_b32_e32 v4, v4, v244, vcc
    v_cndmask_b32_e32 v5, v5, v245, vcc
    v_cndmask_b32_e32 v6, v6, v246, vcc
    v_cndmask_b32_e32 v7, v7, v247, vcc

    // Store packed A regs after the four sets (offset +128 bytes)
    v_add_u32_e32 v31, 32, v30
    buffer_store_dwordx4 v[0:3], v31, s[4:7], 0 offen
    v_add_u32_e32 v31, 16, v31
    buffer_store_dwordx4 v[4:7], v31, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_v_read_dump
    .amdhsa_group_segment_fixed_size 50176
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 20
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 28
    .amdhsa_accum_offset 256
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_v_read_dump
    .symbol: _fwd_fp8_v_read_dump.kd
    .kernarg_segment_size: 20
    .group_segment_fixed_size: 50176
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 28
    .vgpr_count: 240
    .max_flat_workgroup_size: 256
    .args:
      - {.name: out_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: v_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: stride_vh, .size: 4, .offset: 16, .value_kind: by_value}
...
.end_amdgpu_metadata

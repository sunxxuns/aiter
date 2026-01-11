// BF16 Flash Attention Forward kernel for head_dim=128
// Disassembled from fwd_hd128_bf16.co for reference
// This is the production BF16 kernel used as reference for FP8 development

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter19fmha_fwd_hd128_bf16E
.p2align 8
.type _ZN5aiter19fmha_fwd_hd128_bf16E,@function

_ZN5aiter19fmha_fwd_hd128_bf16E:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[20:21], s[0:1], 0x0
    s_load_dwordx2 s[8:9], s[0:1], 0x10
    s_load_dwordx2 s[12:13], s[0:1], 0x20
    s_load_dwordx2 s[16:17], s[0:1], 0x30
    s_load_dwordx2 s[24:25], s[0:1], 0x40
    s_load_dword s28, s[0:1], 0x50
    s_load_dword s30, s[0:1], 0x60
    s_load_dword s50, s[0:1], 0x70
    s_load_dword s31, s[0:1], 0x80
    s_load_dword s32, s[0:1], 0x90
    s_load_dword s33, s[0:1], 0xa0
    s_load_dword s46, s[0:1], 0xb0
    s_load_dword s47, s[0:1], 0xc0
    s_load_dword s48, s[0:1], 0xd0
    s_load_dword s49, s[0:1], 0xe0
    s_load_dword s55, s[0:1], 0xf0
    s_load_dword s56, s[0:1], 0x100
    s_load_dword s7, s[0:1], 0x110
    s_load_dword s90, s[0:1], 0x140
    s_load_dword s76, s[0:1], 0x150
    s_load_dword s77, s[0:1], 0x160
    s_load_dword s78, s[0:1], 0x170
    s_load_dword s79, s[0:1], 0x180
    s_load_dword s80, s[0:1], 0x190
    s_load_dword s81, s[0:1], 0x1a0
    s_load_dwordx2 s[72:73], s[0:1], 0x1b0
    s_load_dwordx2 s[74:75], s[0:1], 0x1c0
    s_load_dword s82, s[0:1], 0x1d0
    s_load_dwordx2 s[84:85], s[0:1], 0x1e0
    s_load_dwordx2 s[86:87], s[0:1], 0x1f0
    v_lshrrev_b32_e32 v1, 10, v0
    v_lshrrev_b32_e32 v2, 10, v1
    v_and_b32_e32 v2, 0x3ff, v2
    v_and_b32_e32 v1, 0x3ff, v1
    v_and_b32_e32 v0, 0x3ff, v0
    v_lshrrev_b32_e32 v3, 6, v0
    v_and_b32_e32 v0, 63, v0
    s_mov_b32 s2, s2
    s_mov_b32 s3, s3
    s_mov_b32 s4, s4
    v_readfirstlane_b32 s5, v3
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s40, s30, s50
    s_mov_b32 s10, s40
    s_mul_i32 s40, s30, s79
    s_mov_b32 s22, s40
    s_mul_i32 s40, s7, s47
    s_mov_b32 s14, s40
    s_mul_i32 s40, s7, s76
    s_mov_b32 s18, s40
    s_mul_i32 s40, s30, 4
    s_mov_b32 s26, s40
    s_mov_b32 s23, 0x20000
    s_mov_b32 s11, 0x20000
    s_mov_b32 s15, 0x20000
    s_mov_b32 s19, 0x20000
    s_mov_b32 s27, 0x20000
    s_and_b32 s21, s21, 0xffff
    s_and_b32 s9, s9, 0xffff
    s_and_b32 s13, s13, 0xffff
    s_and_b32 s17, s17, 0xffff
    s_and_b32 s25, s25, 0xffff
    s_nop 0
    s_nop 0
    s_mov_b32 s60, s3
    s_mov_b32 s61, s46
    v_cvt_f32_u32_e32 v12, s61
    s_sub_i32 s40, 0, s61
    v_rcp_iflag_f32_e32 v12, v12
    s_nop 0
    v_mul_f32_e32 v12, 0x4f7ffffe, v12
    v_cvt_u32_f32_e32 v12, v12
    v_mul_lo_u32 v13, s40, v12
    v_mul_hi_u32 v13, v12, v13
    v_add_u32_e32 v12, v12, v13
    v_mul_hi_u32 v12, s60, v12
    v_mul_lo_u32 v13, v12, s61
    v_sub_u32_e32 v15, s60, v13
    v_add_u32_e32 v14, 1, v12
    v_cmp_le_u32_e32 vcc, s61, v15
    v_subrev_u32_e32 v13, s61, v15
    s_nop 0
    v_cndmask_b32_e32 v12, v12, v14, vcc
    v_cndmask_b32_e32 v15, v15, v13, vcc
    v_add_u32_e32 v13, 1, v12
    v_cmp_le_u32_e32 vcc, s61, v15
    s_nop 1
    v_cndmask_b32_e32 v15, v12, v13, vcc
    s_nop 3
    v_readfirstlane_b32 s62, v15
    s_nop 3
    s_mov_b32 s34, 0
    s_mov_b32 s35, 0
    s_mul_i32 s43, 64, s47
    s_mul_i32 s44, 64, s76
    s_mul_i32 s40, s4, s33
    s_mul_i32 s41, s3, s32
    s_add_u32 s40, s40, s41
    s_add_u32 s8, s40, s8
    s_addc_u32 s9, 0, s9
    s_mul_i32 s40, s4, s81
    s_mul_i32 s41, s3, s80
    s_add_u32 s40, s40, s41
    s_add_u32 s20, s40, s20
    s_addc_u32 s21, 0, s21
    s_mul_i32 s40, s90, s82
    s_mul_i32 s40, s4, s40
    s_mul_i32 s41, s3, s82
    s_nop 0
    s_add_i32 s40, s40, s41
    s_add_u32 s24, s40, s24
    s_addc_u32 s25, 0, s25
    s_mul_i32 s40, s4, s49
    s_mul_i32 s41, s62, s48
    s_add_u32 s40, s40, s41
    s_add_u32 s12, s40, s12
    s_addc_u32 s13, 0, s13
    s_mul_i32 s40, s4, s78
    s_mul_i32 s41, s62, s77
    s_add_u32 s40, s40, s41
    s_add_u32 s16, s40, s16
    s_addc_u32 s17, 0, s17
    s_mov_b32 s52, 0
    s_mov_b32 s53, 64
    s_mov_b32 s36, 0
    s_mov_b32 s29, 0x3fb8aa3b
    v_mov_b32_e32 v27, 0xff800000
    s_lshr_b32 s54, s7, 6
    s_lshl_b32 s54, s54, 6
    v_lshrrev_b32_e32 v12, 5, v0
    v_mul_i32_i24_e32 v26, 4, v12
    s_mov_b32 s38, s7
    s_mov_b32 s39, 0
    v_mov_b32_e32 v13, s29
    v_mov_b32_e32 v12, s28
    v_mul_f32_e32 v12, s29, v12
    v_rcp_f32_e32 v13, v13
    v_mov_b32_e32 v18, 0
    v_mov_b32_e32 v24, 0xff7fffff
    v_mov_b32_e32 v16, 0
    v_mov_b32_e32 v18, 0
    v_readfirstlane_b32 s37, v12
    v_readfirstlane_b32 s45, v13
    v_rcp_f32_e32 v12, v12
    s_nop 1
    v_mul_f32_e32 v12, v24, v12
    v_max_f32_e32 v24, v24, v12
    s_mul_i32 s59, s2, s31
    v_lshrrev_b32_e32 v12, 3, v0
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, s50, v13
    v_lshrrev_b32_e32 v14, 1, v12
    v_mul_i32_i24_e32 v14, s50, v14
    v_mul_i32_i24_e32 v14, 32, v14
    v_and_b32_e32 v12, 7, v0
    v_lshlrev_b32_e32 v12, 4, v12
    s_mul_i32 s40, s5, s50
    s_mul_i32 s40, 2, s40
    s_add_u32 s40, s59, s40
    v_add_u32_e32 v4, s40, v12
    v_add_u32_e32 v4, v13, v4
    v_add_u32_e32 v4, v14, v4
    s_mul_i32 s40, 16, s50
    v_add_u32_e32 v5, s40, v4
    v_add_u32_e32 v6, 0x80, v4
    v_add_u32_e32 v7, s40, v6
    s_mul_i32 s63, 0x408, s5
    s_add_u32 s63, 0x8200, s63
    s_mov_b32 m0, s63
    s_mul_i32 s40, s50, 0x80
    buffer_load_dwordx4 v4, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v4, s40, v4
    v_mov_b32_e32 v96, 0
    v_mov_b32_e32 v97, 0
    v_mov_b32_e32 v98, 0
    v_mov_b32_e32 v99, 0
    v_mov_b32_e32 v100, 0
    v_mov_b32_e32 v101, 0
    v_mov_b32_e32 v102, 0
    v_mov_b32_e32 v103, 0
    v_mov_b32_e32 v104, 0
    v_mov_b32_e32 v105, 0
    v_mov_b32_e32 v106, 0
    v_mov_b32_e32 v107, 0
    v_mov_b32_e32 v108, 0
    v_mov_b32_e32 v109, 0
    v_mov_b32_e32 v110, 0
    v_mov_b32_e32 v111, 0
    buffer_load_dwordx4 v5, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v5, s40, v5
    v_mov_b32_e32 v112, 0
    v_mov_b32_e32 v113, 0
    v_mov_b32_e32 v114, 0
    v_mov_b32_e32 v115, 0
    v_mov_b32_e32 v116, 0
    v_mov_b32_e32 v117, 0
    v_mov_b32_e32 v118, 0
    v_mov_b32_e32 v119, 0
    v_mov_b32_e32 v120, 0
    v_mov_b32_e32 v121, 0
    v_mov_b32_e32 v122, 0
    v_mov_b32_e32 v123, 0
    v_mov_b32_e32 v124, 0
    v_mov_b32_e32 v125, 0
    v_mov_b32_e32 v126, 0
    v_mov_b32_e32 v127, 0
    buffer_load_dwordx4 v6, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v6, s40, v6
    v_mov_b32_e32 v128, 0
    v_mov_b32_e32 v129, 0
    v_mov_b32_e32 v130, 0
    v_mov_b32_e32 v131, 0
    v_mov_b32_e32 v132, 0
    v_mov_b32_e32 v133, 0
    v_mov_b32_e32 v134, 0
    v_mov_b32_e32 v135, 0
    v_mov_b32_e32 v136, 0
    v_mov_b32_e32 v137, 0
    v_mov_b32_e32 v138, 0
    v_mov_b32_e32 v139, 0
    v_mov_b32_e32 v140, 0
    v_mov_b32_e32 v141, 0
    v_mov_b32_e32 v142, 0
    v_mov_b32_e32 v143, 0
    buffer_load_dwordx4 v7, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v7, s40, v7
    v_mov_b32_e32 v144, 0
    v_mov_b32_e32 v145, 0
    v_mov_b32_e32 v146, 0
    v_mov_b32_e32 v147, 0
    v_mov_b32_e32 v148, 0
    v_mov_b32_e32 v149, 0
    v_mov_b32_e32 v150, 0
    v_mov_b32_e32 v151, 0
    v_mov_b32_e32 v152, 0
    v_mov_b32_e32 v153, 0
    v_mov_b32_e32 v154, 0
    v_mov_b32_e32 v155, 0
    v_mov_b32_e32 v156, 0
    v_mov_b32_e32 v157, 0
    v_mov_b32_e32 v158, 0
    v_mov_b32_e32 v159, 0
    buffer_load_dwordx4 v4, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v4, s40, v4
    buffer_load_dwordx4 v5, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v5, s40, v5
    buffer_load_dwordx4 v6, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v6, s40, v6
    buffer_load_dwordx4 v7, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    v_add_u32_e32 v7, s40, v7
    s_cmp_le_u32 s7, 0
    s_cbranch_scc1 label_0EF6
    v_and_b32_e32 v12, 31, v0
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 0x80, v13
    v_lshrrev_b32_e32 v14, 1, v12
    v_mul_i32_i24_e32 v14, 0x408, v14
    v_lshrrev_b32_e32 v12, 5, v0
    v_mul_i32_i24_e32 v12, 16, v12
    v_add_u32_e32 v2, v12, v13
    v_add_u32_e32 v2, v14, v2
    v_add_u32_e32 v2, 0x8200, v2
    s_and_b32 s40, 3, s5
    s_mul_i32 s40, s40, 0x100
    v_add_u32_e32 v2, s40, v2
    s_lshr_b32 s40, s5, 2
    s_mul_i32 s40, s40, 0x8100
    v_add_u32_e32 v2, s40, v2
    v_add_u32_e32 v3, 0x4080, v2
    v_lshrrev_b32_e32 v12, 5, v0
    v_mul_i32_i24_e32 v8, 16, v12
    v_and_b32_e32 v12, 31, v0
    v_lshrrev_b32_e32 v12, 2, v12
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 0x100, v13
    v_lshrrev_b32_e32 v12, 1, v12
    v_mul_i32_i24_e32 v12, 2, v12
    v_mul_i32_i24_e32 v12, 0x410, v12
    v_add_u32_e32 v8, v8, v12
    v_add_u32_e32 v8, v8, v13
    v_and_b32_e32 v12, 3, v0
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 0x80, v13
    v_lshrrev_b32_e32 v12, 1, v12
    v_mul_i32_i24_e32 v12, 0x410, v12
    v_add_u32_e32 v8, v8, v12
    v_add_u32_e32 v8, v8, v13
    v_add_u32_e32 v8, 0, v8
    v_add_u32_e32 v9, 0x4100, v8
    v_lshrrev_b32_e32 v12, 2, v0
    v_and_b32_e32 v12, 3, v12
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 0x80, v13
    v_lshrrev_b32_e32 v12, 1, v12
    v_mul_i32_i24_e32 v12, 0x440, v12
    v_add_u32_e32 v10, v13, v12
    v_and_b32_e32 v12, 3, v0
    v_mul_i32_i24_e32 v12, 8, v12
    v_add_u32_e32 v10, v12, v10
    v_lshrrev_b32_e32 v12, 4, v0
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 32, v13
    v_add_u32_e32 v10, v13, v10
    v_lshrrev_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 0x100, v13
    v_add_u32_e32 v10, v13, v10
    v_add_u32_e32 v10, 0x8200, v10
    v_add_u32_e32 v11, 0x4400, v10
    v_lshrrev_b32_e32 v12, 3, v0
    v_lshrrev_b32_e32 v13, 2, v12
    v_lshlrev_b32_e32 v13, 5, v13
    v_and_b32_e32 v12, 3, v12
    v_lshrrev_b32_e32 v12, 1, v12
    v_lshlrev_b32_e32 v12, 2, v12
    v_add_u32_e32 v13, v13, v12
    v_lshrrev_b32_e32 v12, 3, v0
    v_and_b32_e32 v12, 1, v12
    v_add_u32_e32 v12, v13, v12
    v_mul_i32_i24_e32 v12, s47, v12
    v_and_b32_e32 v4, 7, v0
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v4, v12, v4
    s_and_b32 s40, 1, s5
    s_mul_i32 s40, s40, s47
    s_mul_i32 s40, s40, 2
    s_and_b32 s42, 3, s5
    s_lshr_b32 s41, s42, 1
    s_mul_i32 s41, s41, s47
    s_mul_i32 s41, s41, 8
    s_lshr_b32 s42, s5, 2
    s_mul_i32 s42, s42, s47
    s_mul_i32 s42, s42, 16
    s_add_u32 s40, s41, s40
    s_add_u32 s40, s42, s40
    v_add_u32_e32 v4, s40, v4
    v_add_u32_e32 v5, 0x80, v4
    s_mul_i32 s64, 0x410, s5
    s_add_u32 s64, 0, s64
    s_add_u32 s65, 0x4100, s64
    v_lshrrev_b32_e32 v12, 3, v0
    v_lshrrev_b32_e32 v13, 1, v12
    v_lshlrev_b32_e32 v13, 2, v13
    v_and_b32_e32 v14, 1, v12
    v_add_u32_e32 v13, v14, v13
    v_mul_i32_i24_e32 v13, s76, v13
    v_and_b32_e32 v12, 7, v0
    v_lshlrev_b32_e32 v12, 4, v12
    v_add_u32_e32 v12, v13, v12
    s_and_b32 s40, 1, s5
    s_and_b32 s42, 3, s5
    s_lshr_b32 s41, s42, 1
    s_mul_i32 s40, s40, s76
    s_mul_i32 s40, s40, 2
    s_mul_i32 s41, s41, s76
    s_mul_i32 s41, s41, 16
    s_lshr_b32 s42, s5, 2
    s_mul_i32 s42, s42, s76
    s_mul_i32 s42, s42, 32
    s_add_u32 s40, s41, s40
    s_add_u32 s40, s42, s40
    v_add_u32_e32 v6, s40, v12
    v_add_u32_e32 v7, 0x80, v6
    s_mul_i32 s66, 0x440, s5
    s_add_u32 s66, 0x8200, s66
    s_add_u32 s67, 0x4400, s66
    s_mov_b32 m0, s64
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_add_i32 s34, s43, s34
    s_waitcnt vmcnt(2)
    s_barrier
    ds_read_b64 v[160:161], v2
    ds_read_b64 v[162:163], v2 offset:8
    ds_read_b64 v[164:165], v2 offset:32
    ds_read_b64 v[166:167], v2 offset:40
    ds_read_b64 v[168:169], v2 offset:64
    ds_read_b64 v[170:171], v2 offset:72
    ds_read_b64 v[172:173], v2 offset:96
    ds_read_b64 v[174:175], v2 offset:104
    ds_read_b64 v[176:177], v3
    ds_read_b64 v[178:179], v3 offset:8
    ds_read_b64 v[180:181], v3 offset:32
    ds_read_b64 v[182:183], v3 offset:40
    ds_read_b64 v[184:185], v3 offset:64
    ds_read_b64 v[186:187], v3 offset:72
    ds_read_b64 v[188:189], v3 offset:96
    ds_read_b64 v[190:191], v3 offset:104
    s_waitcnt vmcnt(0)
    s_barrier
    ds_read_b128 v[192:195], v8
    ds_read_b128 v[208:211], v8 offset:512
    ds_read_b128 v[196:199], v8 offset:32
    ds_read_b128 v[212:215], v8 offset:544
    ds_read_b128 v[200:203], v8 offset:64
    ds_read_b128 v[216:219], v8 offset:576
    ds_read_b128 v[204:207], v8 offset:96
    ds_read_b128 v[220:223], v8 offset:608
    ds_read_b128 v[224:227], v8 offset:8320
    ds_read_b128 v[240:243], v8 offset:8832
    ds_read_b128 v[228:231], v8 offset:8352
    ds_read_b128 v[244:247], v8 offset:8864
    ds_read_b128 v[232:235], v8 offset:8384
    ds_read_b128 v[248:251], v8 offset:8896
    ds_read_b128 v[236:239], v8 offset:8416
    ds_read_b128 v[252:255], v8 offset:8928
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_mov_b32 m0, s65
    v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[160:163], 0
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[164:167], v[32:47]
    v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[168:171], v[32:47]
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_mov_b32 m0, s66
    v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[172:175], v[32:47]
    v_mfma_f32_32x32x16_bf16 v[32:47], v[224:227], v[176:179], v[32:47]
    buffer_load_dwordx4 v6, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    v_mfma_f32_32x32x16_bf16 v[32:47], v[228:231], v[180:183], v[32:47]
    v_mfma_f32_32x32x16_bf16 v[32:47], v[232:235], v[184:187], v[32:47]
    buffer_load_dwordx4 v7, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    v_mfma_f32_32x32x16_bf16 v[32:47], v[236:239], v[188:191], v[32:47]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[208:211], v[160:163], 0
    v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[164:167], v[48:63]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[216:219], v[168:171], v[48:63]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[172:175], v[48:63]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[240:243], v[176:179], v[48:63]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[244:247], v[180:183], v[48:63]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[248:251], v[184:187], v[48:63]
    v_mfma_f32_32x32x16_bf16 v[48:63], v[252:255], v[188:191], v[48:63]
    s_add_i32 s34, s43, s34
    s_add_i32 s35, s44, s35
    s_cmp_lt_i32 s52, s54
    s_cbranch_scc1 label_02DE
    s_sub_i32 s40, s7, s52
    v_sub_i32 v12, s40, v26
    v_cmp_lt_i32_e64 s[68:69], 0, v12
    v_cmp_lt_i32_e64 s[70:71], 1, v12
    v_cndmask_b32_e64 v32, v27, v32, s[68:69]
    v_cndmask_b32_e64 v33, v27, v33, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 2, v12
    v_cmp_lt_i32_e64 s[70:71], 3, v12
    v_cndmask_b32_e64 v34, v27, v34, s[68:69]
    v_cndmask_b32_e64 v35, v27, v35, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 8, v12
    v_cmp_lt_i32_e64 s[70:71], 9, v12
    v_cndmask_b32_e64 v36, v27, v36, s[68:69]
    v_cndmask_b32_e64 v37, v27, v37, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 10, v12
    v_cmp_lt_i32_e64 s[70:71], 11, v12
    v_cndmask_b32_e64 v38, v27, v38, s[68:69]
    v_cndmask_b32_e64 v39, v27, v39, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 16, v12
    v_cmp_lt_i32_e64 s[70:71], 17, v12
    v_cndmask_b32_e64 v40, v27, v40, s[68:69]
    v_cndmask_b32_e64 v41, v27, v41, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 18, v12
    v_cmp_lt_i32_e64 s[70:71], 19, v12
    v_cndmask_b32_e64 v42, v27, v42, s[68:69]
    v_cndmask_b32_e64 v43, v27, v43, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 24, v12
    v_cmp_lt_i32_e64 s[70:71], 25, v12
    v_cndmask_b32_e64 v44, v27, v44, s[68:69]
    v_cndmask_b32_e64 v45, v27, v45, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 26, v12
    v_cmp_lt_i32_e64 s[70:71], 27, v12
    v_cndmask_b32_e64 v46, v27, v46, s[68:69]
    v_cndmask_b32_e64 v47, v27, v47, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 32, v12
    v_cmp_lt_i32_e64 s[70:71], 33, v12
    v_cndmask_b32_e64 v48, v27, v48, s[68:69]
    v_cndmask_b32_e64 v49, v27, v49, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 34, v12
    v_cmp_lt_i32_e64 s[70:71], 35, v12
    v_cndmask_b32_e64 v50, v27, v50, s[68:69]
    v_cndmask_b32_e64 v51, v27, v51, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 40, v12
    v_cmp_lt_i32_e64 s[70:71], 41, v12
    v_cndmask_b32_e64 v52, v27, v52, s[68:69]
    v_cndmask_b32_e64 v53, v27, v53, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 42, v12
    v_cmp_lt_i32_e64 s[70:71], 43, v12
    v_cndmask_b32_e64 v54, v27, v54, s[68:69]
    v_cndmask_b32_e64 v55, v27, v55, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 48, v12
    v_cmp_lt_i32_e64 s[70:71], 49, v12
    v_cndmask_b32_e64 v56, v27, v56, s[68:69]
    v_cndmask_b32_e64 v57, v27, v57, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 50, v12
    v_cmp_lt_i32_e64 s[70:71], 51, v12
    v_cndmask_b32_e64 v58, v27, v58, s[68:69]
    v_cndmask_b32_e64 v59, v27, v59, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 56, v12
    v_cmp_lt_i32_e64 s[70:71], 57, v12
    v_cndmask_b32_e64 v60, v27, v60, s[68:69]
    v_cndmask_b32_e64 v61, v27, v61, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 58, v12
    v_cmp_lt_i32_e64 s[70:71], 59, v12
    v_cndmask_b32_e64 v62, v27, v62, s[68:69]
    v_cndmask_b32_e64 v63, v27, v63, s[70:71]
    v_mov_b32_e32 v21, v24
    v_max3_f32 v21, v32, v33, v21
    v_max3_f32 v21, v34, v35, v21
    v_max3_f32 v21, v36, v37, v21
    v_max3_f32 v21, v38, v39, v21
    v_max3_f32 v21, v40, v41, v21
    v_max3_f32 v21, v42, v43, v21
    v_max3_f32 v21, v44, v45, v21
    v_max3_f32 v21, v46, v47, v21
    v_max3_f32 v21, v48, v49, v21
    v_max3_f32 v21, v50, v51, v21
    v_max3_f32 v21, v52, v53, v21
    v_max3_f32 v21, v54, v55, v21
    v_max3_f32 v21, v56, v57, v21
    v_max3_f32 v21, v58, v59, v21
    v_max3_f32 v21, v60, v61, v21
    v_max3_f32 v21, v62, v63, v21
    v_mov_b32_e32 v20, v21
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21
    v_mov_b32_e32 v16, 0
    v_mov_b32_e32 v24, v21
    v_mul_f32_e32 v23, s37, v21
    v_mul_f32_e32 v16, s37, v16
    v_exp_f32_e32 v16, v16
    v_fma_f32 v32, v32, s37, -v23
    v_fma_f32 v33, v33, s37, -v23
    v_fma_f32 v34, v34, s37, -v23
    v_fma_f32 v35, v35, s37, -v23
    v_fma_f32 v36, v36, s37, -v23
    v_fma_f32 v37, v37, s37, -v23
    v_fma_f32 v38, v38, s37, -v23
    v_fma_f32 v39, v39, s37, -v23
    v_fma_f32 v40, v40, s37, -v23
    v_fma_f32 v41, v41, s37, -v23
    v_fma_f32 v42, v42, s37, -v23
    v_fma_f32 v43, v43, s37, -v23
    v_fma_f32 v44, v44, s37, -v23
    v_fma_f32 v45, v45, s37, -v23
    v_fma_f32 v46, v46, s37, -v23
    v_fma_f32 v47, v47, s37, -v23
    v_fma_f32 v48, v48, s37, -v23
    v_fma_f32 v49, v49, s37, -v23
    v_fma_f32 v50, v50, s37, -v23
    v_fma_f32 v51, v51, s37, -v23
    v_fma_f32 v52, v52, s37, -v23
    v_fma_f32 v53, v53, s37, -v23
    v_fma_f32 v54, v54, s37, -v23
    v_fma_f32 v55, v55, s37, -v23
    v_fma_f32 v56, v56, s37, -v23
    v_fma_f32 v57, v57, s37, -v23
    v_fma_f32 v58, v58, s37, -v23
    v_fma_f32 v59, v59, s37, -v23
    v_fma_f32 v60, v60, s37, -v23
    v_fma_f32 v61, v61, s37, -v23
    v_fma_f32 v62, v62, s37, -v23
    v_fma_f32 v63, v63, s37, -v23
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
    v_exp_f32_e32 v48, v48
    v_exp_f32_e32 v49, v49
    v_mov_b32_e32 v17, v16
    v_mul_f32_e32 v110, v16, v110
    v_mul_f32_e32 v111, v16, v111
    v_pk_mul_f32 v[112:113], v[16:17], v[112:113]
    v_pk_mul_f32 v[114:115], v[16:17], v[114:115]
    v_pk_mul_f32 v[116:117], v[16:17], v[116:117]
    v_pk_mul_f32 v[118:119], v[16:17], v[118:119]
    v_pk_mul_f32 v[120:121], v[16:17], v[120:121]
    v_pk_mul_f32 v[122:123], v[16:17], v[122:123]
    v_pk_mul_f32 v[124:125], v[16:17], v[124:125]
    v_pk_mul_f32 v[126:127], v[16:17], v[126:127]
    v_pk_mul_f32 v[128:129], v[16:17], v[128:129]
    v_pk_mul_f32 v[130:131], v[16:17], v[130:131]
    v_pk_mul_f32 v[132:133], v[16:17], v[132:133]
    v_pk_mul_f32 v[134:135], v[16:17], v[134:135]
    v_pk_mul_f32 v[136:137], v[16:17], v[136:137]
    v_pk_mul_f32 v[138:139], v[16:17], v[138:139]
    v_pk_mul_f32 v[140:141], v[16:17], v[140:141]
    v_pk_mul_f32 v[142:143], v[16:17], v[142:143]
    v_pk_mul_f32 v[144:145], v[16:17], v[144:145]
    v_pk_mul_f32 v[146:147], v[16:17], v[146:147]
    v_pk_mul_f32 v[148:149], v[16:17], v[148:149]
    v_pk_mul_f32 v[150:151], v[16:17], v[150:151]
    v_pk_mul_f32 v[152:153], v[16:17], v[152:153]
    v_pk_mul_f32 v[154:155], v[16:17], v[154:155]
    v_pk_mul_f32 v[156:157], v[16:17], v[156:157]
    v_pk_mul_f32 v[158:159], v[16:17], v[158:159]
    s_addk_i32 s39, 0x40
    s_add_i32 s52, s52, s53
    s_mov_b32 m0, s64
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_add_i32 s34, s43, s34
    s_waitcnt vmcnt(4)
    s_barrier
    s_cmp_lt_i32 s39, s38
    s_cbranch_scc0 label_0D6A
    s_cmp_lt_i32 s5, 4
    s_cbranch_scc0 label_08A3
    s_mov_b32 m0, s67
    buffer_load_dwordx4 v6, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    buffer_load_dwordx4 v7, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    s_add_i32 s35, s44, s35
    ds_read_b128 v[192:195], v9
    ds_read_b128 v[208:211], v9 offset:512
    ds_read_b128 v[196:199], v9 offset:32
    ds_read_b128 v[212:215], v9 offset:544
    ds_read_b128 v[200:203], v9 offset:64
    ds_read_b128 v[216:219], v9 offset:576
    ds_read_b128 v[204:207], v9 offset:96
    ds_read_b128 v[220:223], v9 offset:608
    ds_read_b128 v[224:227], v9 offset:8320
    ds_read_b128 v[240:243], v9 offset:8832
    ds_read_b128 v[228:231], v9 offset:8352
    ds_read_b128 v[244:247], v9 offset:8864
    ds_read_b128 v[232:235], v9 offset:8384
    ds_read_b128 v[248:251], v9 offset:8896
    ds_read_b128 v[236:239], v9 offset:8416
    ds_read_b128 v[252:255], v9 offset:8928
    s_setprio 0
    s_barrier
    s_waitcnt lgkmcnt(0)
    v_nop
    v_mfma_f32_32x32x16_bf16 v[64:79], v[192:195], v[160:163], 0
    v_exp_f32_e32 v50, v50
    v_exp_f32_e32 v51, v51
    v_exp_f32_e32 v52, v52
    v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[164:167], v[64:79]
    v_exp_f32_e32 v53, v53
    v_exp_f32_e32 v54, v54
    v_exp_f32_e32 v55, v55
    v_mfma_f32_32x32x16_bf16 v[64:79], v[200:203], v[168:171], v[64:79]
    v_exp_f32_e32 v56, v56
    v_exp_f32_e32 v57, v57
    v_exp_f32_e32 v58, v58
    v_mfma_f32_32x32x16_bf16 v[64:79], v[204:207], v[172:175], v[64:79]
    v_exp_f32_e32 v59, v59
    v_exp_f32_e32 v60, v60
    v_exp_f32_e32 v61, v61
    v_mfma_f32_32x32x16_bf16 v[64:79], v[224:227], v[176:179], v[64:79]
    v_exp_f32_e32 v62, v62
    v_exp_f32_e32 v63, v63
    v_mul_f32_e32 v18, v16, v18
    v_add_f32_e32 v19, v33, v32
    v_mfma_f32_32x32x16_bf16 v[64:79], v[228:231], v[180:183], v[64:79]
    v_add_f32_e32 v19, v34, v19
    v_add_f32_e32 v19, v35, v19
    v_add_f32_e32 v19, v36, v19
    v_add_f32_e32 v19, v37, v19
    v_add_f32_e32 v19, v38, v19
    v_add_f32_e32 v19, v39, v19
    v_mfma_f32_32x32x16_bf16 v[64:79], v[232:235], v[184:187], v[64:79]
    v_add_f32_e32 v19, v40, v19
    v_add_f32_e32 v19, v41, v19
    v_add_f32_e32 v19, v42, v19
    v_add_f32_e32 v19, v43, v19
    v_add_f32_e32 v19, v44, v19
    v_add_f32_e32 v19, v45, v19
    v_mfma_f32_32x32x16_bf16 v[64:79], v[236:239], v[188:191], v[64:79]
    v_add_f32_e32 v19, v46, v19
    v_add_f32_e32 v19, v47, v19
    v_add_f32_e32 v19, v48, v19
    v_add_f32_e32 v19, v49, v19
    v_add_f32_e32 v19, v50, v19
    v_add_f32_e32 v19, v51, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[208:211], v[160:163], 0
    v_add_f32_e32 v19, v52, v19
    v_add_f32_e32 v19, v53, v19
    v_add_f32_e32 v19, v54, v19
    v_add_f32_e32 v19, v55, v19
    v_add_f32_e32 v19, v56, v19
    v_add_f32_e32 v19, v57, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[212:215], v[164:167], v[80:95]
    v_add_f32_e32 v19, v58, v19
    v_add_f32_e32 v19, v59, v19
    v_add_f32_e32 v19, v60, v19
    v_add_f32_e32 v19, v61, v19
    v_add_f32_e32 v19, v62, v19
    v_add_f32_e32 v19, v63, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[216:219], v[168:171], v[80:95]
    v_mov_b32_e32 v20, v19
    v_mul_f32_e32 v96, v16, v96
    v_mul_f32_e32 v97, v16, v97
    v_permlane32_swap_b32_e32 v20, v19
    v_add_f32_e32 v20, v20, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[220:223], v[172:175], v[80:95]
    v_add_f32_e32 v18, v20, v18
    v_mul_f32_e32 v98, v16, v98
    v_mul_f32_e32 v99, v16, v99
    v_mul_f32_e32 v100, v16, v100
    v_mul_f32_e32 v101, v16, v101
    v_mul_f32_e32 v102, v16, v102
    v_mfma_f32_32x32x16_bf16 v[80:95], v[240:243], v[176:179], v[80:95]
    v_mul_f32_e32 v103, v16, v103
    v_mul_f32_e32 v104, v16, v104
    v_mul_f32_e32 v105, v16, v105
    v_mul_f32_e32 v106, v16, v106
    v_mul_f32_e32 v107, v16, v107
    v_mul_f32_e32 v108, v16, v108
    v_mfma_f32_32x32x16_bf16 v[80:95], v[244:247], v[180:183], v[80:95]
    v_mul_f32_e32 v109, v16, v109
    v_cvt_pk_bf16_f32 v32, v32, v33
    v_cvt_pk_bf16_f32 v33, v34, v35
    v_cvt_pk_bf16_f32 v34, v36, v37
    v_cvt_pk_bf16_f32 v35, v38, v39
    v_cvt_pk_bf16_f32 v36, v40, v41
    v_mfma_f32_32x32x16_bf16 v[80:95], v[248:251], v[184:187], v[80:95]
    v_cvt_pk_bf16_f32 v37, v42, v43
    v_cvt_pk_bf16_f32 v38, v44, v45
    v_cvt_pk_bf16_f32 v39, v46, v47
    v_cvt_pk_bf16_f32 v40, v48, v49
    v_cvt_pk_bf16_f32 v41, v50, v51
    v_cvt_pk_bf16_f32 v42, v52, v53
    v_mfma_f32_32x32x16_bf16 v[80:95], v[252:255], v[188:191], v[80:95]
    v_cvt_pk_bf16_f32 v43, v54, v55
    v_cvt_pk_bf16_f32 v44, v56, v57
    v_cvt_pk_bf16_f32 v45, v58, v59
    v_cvt_pk_bf16_f32 v46, v60, v61
    v_cvt_pk_bf16_f32 v47, v62, v63
    s_waitcnt vmcnt(4)
    s_barrier
    s_mov_b32 m0, s65
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_add_i32 s34, s43, s34
    s_nop 0
    s_add_u32 s40, 0x100, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s43, s43, 0
    ds_read_b64_tr_b16 v[192:193], v10
    ds_read_b64_tr_b16 v[194:195], v10 offset:512
    ds_read_b64_tr_b16 v[208:209], v10 offset:64
    ds_read_b64_tr_b16 v[210:211], v10 offset:576
    ds_read_b64_tr_b16 v[196:197], v10 offset:2176
    ds_read_b64_tr_b16 v[198:199], v10 offset:2688
    ds_read_b64_tr_b16 v[212:213], v10 offset:2240
    ds_read_b64_tr_b16 v[214:215], v10 offset:2752
    ds_read_b64_tr_b16 v[200:201], v10 offset:4352
    ds_read_b64_tr_b16 v[202:203], v10 offset:4864
    ds_read_b64_tr_b16 v[216:217], v10 offset:4416
    ds_read_b64_tr_b16 v[218:219], v10 offset:4928
    ds_read_b64_tr_b16 v[204:205], v10 offset:6528
    ds_read_b64_tr_b16 v[206:207], v10 offset:7040
    ds_read_b64_tr_b16 v[220:221], v10 offset:6592
    ds_read_b64_tr_b16 v[222:223], v10 offset:7104
    ds_read_b64_tr_b16 v[224:225], v10 offset:8704
    ds_read_b64_tr_b16 v[226:227], v10 offset:9216
    ds_read_b64_tr_b16 v[240:241], v10 offset:8768
    ds_read_b64_tr_b16 v[242:243], v10 offset:9280
    ds_read_b64_tr_b16 v[228:229], v10 offset:10880
    ds_read_b64_tr_b16 v[230:231], v10 offset:11392
    ds_read_b64_tr_b16 v[244:245], v10 offset:10944
    ds_read_b64_tr_b16 v[246:247], v10 offset:11456
    ds_read_b64_tr_b16 v[232:233], v10 offset:13056
    ds_read_b64_tr_b16 v[234:235], v10 offset:13568
    ds_read_b64_tr_b16 v[248:249], v10 offset:13120
    ds_read_b64_tr_b16 v[250:251], v10 offset:13632
    ds_read_b64_tr_b16 v[236:237], v10 offset:15232
    ds_read_b64_tr_b16 v[238:239], v10 offset:15744
    ds_read_b64_tr_b16 v[252:253], v10 offset:15296
    ds_read_b64_tr_b16 v[254:255], v10 offset:15808
    s_nop 0
    s_cmp_lt_i32 s52, s54
    s_cbranch_scc1 label_0526
    s_sub_i32 s40, s7, s52
    v_sub_i32 v12, s40, v26
    v_cmp_lt_i32_e64 s[68:69], 0, v12
    v_cmp_lt_i32_e64 s[70:71], 1, v12
    v_cndmask_b32_e64 v64, v27, v64, s[68:69]
    v_cndmask_b32_e64 v65, v27, v65, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 2, v12
    v_cmp_lt_i32_e64 s[70:71], 3, v12
    v_cndmask_b32_e64 v66, v27, v66, s[68:69]
    v_cndmask_b32_e64 v67, v27, v67, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 8, v12
    v_cmp_lt_i32_e64 s[70:71], 9, v12
    v_cndmask_b32_e64 v68, v27, v68, s[68:69]
    v_cndmask_b32_e64 v69, v27, v69, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 10, v12
    v_cmp_lt_i32_e64 s[70:71], 11, v12
    v_cndmask_b32_e64 v70, v27, v70, s[68:69]
    v_cndmask_b32_e64 v71, v27, v71, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 16, v12
    v_cmp_lt_i32_e64 s[70:71], 17, v12
    v_cndmask_b32_e64 v72, v27, v72, s[68:69]
    v_cndmask_b32_e64 v73, v27, v73, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 18, v12
    v_cmp_lt_i32_e64 s[70:71], 19, v12
    v_cndmask_b32_e64 v74, v27, v74, s[68:69]
    v_cndmask_b32_e64 v75, v27, v75, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 24, v12
    v_cmp_lt_i32_e64 s[70:71], 25, v12
    v_cndmask_b32_e64 v76, v27, v76, s[68:69]
    v_cndmask_b32_e64 v77, v27, v77, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 26, v12
    v_cmp_lt_i32_e64 s[70:71], 27, v12
    v_cndmask_b32_e64 v78, v27, v78, s[68:69]
    v_cndmask_b32_e64 v79, v27, v79, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 32, v12
    v_cmp_lt_i32_e64 s[70:71], 33, v12
    v_cndmask_b32_e64 v80, v27, v80, s[68:69]
    v_cndmask_b32_e64 v81, v27, v81, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 34, v12
    v_cmp_lt_i32_e64 s[70:71], 35, v12
    v_cndmask_b32_e64 v82, v27, v82, s[68:69]
    v_cndmask_b32_e64 v83, v27, v83, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 40, v12
    v_cmp_lt_i32_e64 s[70:71], 41, v12
    v_cndmask_b32_e64 v84, v27, v84, s[68:69]
    v_cndmask_b32_e64 v85, v27, v85, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 42, v12
    v_cmp_lt_i32_e64 s[70:71], 43, v12
    v_cndmask_b32_e64 v86, v27, v86, s[68:69]
    v_cndmask_b32_e64 v87, v27, v87, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 48, v12
    v_cmp_lt_i32_e64 s[70:71], 49, v12
    v_cndmask_b32_e64 v88, v27, v88, s[68:69]
    v_cndmask_b32_e64 v89, v27, v89, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 50, v12
    v_cmp_lt_i32_e64 s[70:71], 51, v12
    v_cndmask_b32_e64 v90, v27, v90, s[68:69]
    v_cndmask_b32_e64 v91, v27, v91, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 56, v12
    v_cmp_lt_i32_e64 s[70:71], 57, v12
    v_cndmask_b32_e64 v92, v27, v92, s[68:69]
    v_cndmask_b32_e64 v93, v27, v93, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 58, v12
    v_cmp_lt_i32_e64 s[70:71], 59, v12
    v_cndmask_b32_e64 v94, v27, v94, s[68:69]
    v_cndmask_b32_e64 v95, v27, v95, s[70:71]
    s_nop 0
    s_nop 0
    s_nop 0
    s_waitcnt lgkmcnt(0)
    s_barrier
    v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[32:35], v[96:111]
    v_mov_b32_e32 v21, v24
    v_max3_f32 v21, v64, v65, v21
    v_max3_f32 v21, v66, v67, v21
    v_max3_f32 v21, v68, v69, v21
    v_max3_f32 v21, v70, v71, v21
    v_max3_f32 v21, v72, v73, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[196:199], v[36:39], v[96:111]
    v_max3_f32 v21, v74, v75, v21
    v_max3_f32 v21, v76, v77, v21
    v_max3_f32 v21, v78, v79, v21
    v_max3_f32 v21, v80, v81, v21
    v_max3_f32 v21, v82, v83, v21
    v_max3_f32 v21, v84, v85, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[200:203], v[40:43], v[96:111]
    v_max3_f32 v21, v86, v87, v21
    v_max3_f32 v21, v88, v89, v21
    v_max3_f32 v21, v90, v91, v21
    v_max3_f32 v21, v92, v93, v21
    v_max3_f32 v21, v94, v95, v21
    v_mov_b32_e32 v20, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[204:207], v[44:47], v[96:111]
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21
    v_sub_f32_e32 v16, v24, v21
    v_mfma_f32_32x32x16_bf16 v[112:127], v[208:211], v[32:35], v[112:127]
    v_mov_b32_e32 v24, v21
    v_mul_f32_e32 v23, s37, v21
    v_mul_f32_e32 v16, s37, v16
    v_exp_f32_e32 v16, v16
    v_fma_f32 v64, v64, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[212:215], v[36:39], v[112:127]
    v_fma_f32 v65, v65, s37, -v23
    v_fma_f32 v66, v66, s37, -v23
    v_fma_f32 v67, v67, s37, -v23
    v_fma_f32 v68, v68, s37, -v23
    v_fma_f32 v69, v69, s37, -v23
    v_fma_f32 v70, v70, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[216:219], v[40:43], v[112:127]
    v_fma_f32 v71, v71, s37, -v23
    v_fma_f32 v72, v72, s37, -v23
    v_fma_f32 v73, v73, s37, -v23
    v_fma_f32 v74, v74, s37, -v23
    v_fma_f32 v75, v75, s37, -v23
    v_fma_f32 v76, v76, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[44:47], v[112:127]
    v_fma_f32 v77, v77, s37, -v23
    v_fma_f32 v78, v78, s37, -v23
    v_fma_f32 v79, v79, s37, -v23
    v_fma_f32 v80, v80, s37, -v23
    v_fma_f32 v81, v81, s37, -v23
    v_fma_f32 v82, v82, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[32:35], v[128:143]
    v_fma_f32 v83, v83, s37, -v23
    v_fma_f32 v84, v84, s37, -v23
    v_fma_f32 v85, v85, s37, -v23
    v_fma_f32 v86, v86, s37, -v23
    v_fma_f32 v87, v87, s37, -v23
    v_fma_f32 v88, v88, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[36:39], v[128:143]
    v_fma_f32 v89, v89, s37, -v23
    v_fma_f32 v90, v90, s37, -v23
    v_fma_f32 v91, v91, s37, -v23
    v_fma_f32 v92, v92, s37, -v23
    v_fma_f32 v93, v93, s37, -v23
    v_fma_f32 v94, v94, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[40:43], v[128:143]
    v_fma_f32 v95, v95, s37, -v23
    v_exp_f32_e32 v64, v64
    v_exp_f32_e32 v65, v65
    v_exp_f32_e32 v66, v66
    v_mfma_f32_32x32x16_bf16 v[128:143], v[236:239], v[44:47], v[128:143]
    v_exp_f32_e32 v67, v67
    v_exp_f32_e32 v68, v68
    v_exp_f32_e32 v69, v69
    v_mfma_f32_32x32x16_bf16 v[144:159], v[240:243], v[32:35], v[144:159]
    v_exp_f32_e32 v70, v70
    v_exp_f32_e32 v71, v71
    v_exp_f32_e32 v72, v72
    v_mfma_f32_32x32x16_bf16 v[144:159], v[244:247], v[36:39], v[144:159]
    v_exp_f32_e32 v73, v73
    v_exp_f32_e32 v74, v74
    v_exp_f32_e32 v75, v75
    v_mfma_f32_32x32x16_bf16 v[144:159], v[248:251], v[40:43], v[144:159]
    v_exp_f32_e32 v76, v76
    v_exp_f32_e32 v77, v77
    v_exp_f32_e32 v78, v78
    v_mfma_f32_32x32x16_bf16 v[144:159], v[252:255], v[44:47], v[144:159]
    v_exp_f32_e32 v79, v79
    v_exp_f32_e32 v80, v80
    v_exp_f32_e32 v81, v81
    s_nop 0
    v_mov_b32_e32 v17, v16
    v_mul_f32_e32 v110, v16, v110
    v_mul_f32_e32 v111, v16, v111
    v_pk_mul_f32 v[112:113], v[16:17], v[112:113]
    v_pk_mul_f32 v[114:115], v[16:17], v[114:115]
    v_pk_mul_f32 v[116:117], v[16:17], v[116:117]
    v_pk_mul_f32 v[118:119], v[16:17], v[118:119]
    v_pk_mul_f32 v[120:121], v[16:17], v[120:121]
    v_pk_mul_f32 v[122:123], v[16:17], v[122:123]
    v_pk_mul_f32 v[124:125], v[16:17], v[124:125]
    v_pk_mul_f32 v[126:127], v[16:17], v[126:127]
    v_pk_mul_f32 v[128:129], v[16:17], v[128:129]
    v_pk_mul_f32 v[130:131], v[16:17], v[130:131]
    v_pk_mul_f32 v[132:133], v[16:17], v[132:133]
    v_pk_mul_f32 v[134:135], v[16:17], v[134:135]
    v_pk_mul_f32 v[136:137], v[16:17], v[136:137]
    v_pk_mul_f32 v[138:139], v[16:17], v[138:139]
    v_pk_mul_f32 v[140:141], v[16:17], v[140:141]
    v_pk_mul_f32 v[142:143], v[16:17], v[142:143]
    v_pk_mul_f32 v[144:145], v[16:17], v[144:145]
    v_pk_mul_f32 v[146:147], v[16:17], v[146:147]
    v_pk_mul_f32 v[148:149], v[16:17], v[148:149]
    v_pk_mul_f32 v[150:151], v[16:17], v[150:151]
    v_pk_mul_f32 v[152:153], v[16:17], v[152:153]
    v_pk_mul_f32 v[154:155], v[16:17], v[154:155]
    v_pk_mul_f32 v[156:157], v[16:17], v[156:157]
    v_pk_mul_f32 v[158:159], v[16:17], v[158:159]
    s_waitcnt vmcnt(4)
    s_barrier
    s_nop 15
    s_nop 7
    s_mov_b32 m0, s66
    buffer_load_dwordx4 v6, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    buffer_load_dwordx4 v7, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    s_add_i32 s35, s44, s35
    s_nop 0
    s_add_u32 s40, 0xc0, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s44, s44, 0
    s_nop 15
    s_nop 15
    s_nop 7
    ds_read_b128 v[192:195], v8
    ds_read_b128 v[208:211], v8 offset:512
    ds_read_b128 v[196:199], v8 offset:32
    ds_read_b128 v[212:215], v8 offset:544
    ds_read_b128 v[200:203], v8 offset:64
    ds_read_b128 v[216:219], v8 offset:576
    ds_read_b128 v[204:207], v8 offset:96
    ds_read_b128 v[220:223], v8 offset:608
    ds_read_b128 v[224:227], v8 offset:8320
    ds_read_b128 v[240:243], v8 offset:8832
    ds_read_b128 v[228:231], v8 offset:8352
    ds_read_b128 v[244:247], v8 offset:8864
    ds_read_b128 v[232:235], v8 offset:8384
    ds_read_b128 v[248:251], v8 offset:8896
    ds_read_b128 v[236:239], v8 offset:8416
    ds_read_b128 v[252:255], v8 offset:8928
    s_add_i32 s52, s52, s53
    s_addk_i32 s39, 0x40
    s_cmp_lt_i32 s39, s38
    s_cbranch_scc0 label_0D6A
    s_waitcnt lgkmcnt(0)
    v_nop
    v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[160:163], 0
    v_exp_f32_e32 v82, v82
    v_exp_f32_e32 v83, v83
    v_exp_f32_e32 v84, v84
    v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[164:167], v[32:47]
    v_exp_f32_e32 v85, v85
    v_exp_f32_e32 v86, v86
    v_exp_f32_e32 v87, v87
    v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[168:171], v[32:47]
    v_exp_f32_e32 v88, v88
    v_exp_f32_e32 v89, v89
    v_exp_f32_e32 v90, v90
    v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[172:175], v[32:47]
    v_exp_f32_e32 v91, v91
    v_exp_f32_e32 v92, v92
    v_exp_f32_e32 v93, v93
    v_mfma_f32_32x32x16_bf16 v[32:47], v[224:227], v[176:179], v[32:47]
    v_exp_f32_e32 v94, v94
    v_exp_f32_e32 v95, v95
    v_mul_f32_e32 v18, v16, v18
    v_add_f32_e32 v19, v65, v64
    v_mfma_f32_32x32x16_bf16 v[32:47], v[228:231], v[180:183], v[32:47]
    v_add_f32_e32 v19, v66, v19
    v_add_f32_e32 v19, v67, v19
    v_add_f32_e32 v19, v68, v19
    v_add_f32_e32 v19, v69, v19
    v_add_f32_e32 v19, v70, v19
    v_add_f32_e32 v19, v71, v19
    v_mfma_f32_32x32x16_bf16 v[32:47], v[232:235], v[184:187], v[32:47]
    v_add_f32_e32 v19, v72, v19
    v_add_f32_e32 v19, v73, v19
    v_add_f32_e32 v19, v74, v19
    v_add_f32_e32 v19, v75, v19
    v_add_f32_e32 v19, v76, v19
    v_add_f32_e32 v19, v77, v19
    v_mfma_f32_32x32x16_bf16 v[32:47], v[236:239], v[188:191], v[32:47]
    v_add_f32_e32 v19, v78, v19
    v_add_f32_e32 v19, v79, v19
    v_add_f32_e32 v19, v80, v19
    v_add_f32_e32 v19, v81, v19
    v_add_f32_e32 v19, v82, v19
    v_add_f32_e32 v19, v83, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[208:211], v[160:163], 0
    v_add_f32_e32 v19, v84, v19
    v_add_f32_e32 v19, v85, v19
    v_add_f32_e32 v19, v86, v19
    v_add_f32_e32 v19, v87, v19
    v_add_f32_e32 v19, v88, v19
    v_add_f32_e32 v19, v89, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[164:167], v[48:63]
    v_add_f32_e32 v19, v90, v19
    v_add_f32_e32 v19, v91, v19
    v_add_f32_e32 v19, v92, v19
    v_add_f32_e32 v19, v93, v19
    v_add_f32_e32 v19, v94, v19
    v_add_f32_e32 v19, v95, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[216:219], v[168:171], v[48:63]
    v_mov_b32_e32 v20, v19
    v_mul_f32_e32 v96, v16, v96
    v_mul_f32_e32 v97, v16, v97
    v_permlane32_swap_b32_e32 v20, v19
    v_add_f32_e32 v20, v20, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[172:175], v[48:63]
    v_add_f32_e32 v18, v20, v18
    v_mul_f32_e32 v98, v16, v98
    v_mul_f32_e32 v99, v16, v99
    v_mul_f32_e32 v100, v16, v100
    v_mul_f32_e32 v101, v16, v101
    v_mul_f32_e32 v102, v16, v102
    v_mfma_f32_32x32x16_bf16 v[48:63], v[240:243], v[176:179], v[48:63]
    v_mul_f32_e32 v103, v16, v103
    v_mul_f32_e32 v104, v16, v104
    v_mul_f32_e32 v105, v16, v105
    v_mul_f32_e32 v106, v16, v106
    v_mul_f32_e32 v107, v16, v107
    v_mul_f32_e32 v108, v16, v108
    v_mfma_f32_32x32x16_bf16 v[48:63], v[244:247], v[180:183], v[48:63]
    v_mul_f32_e32 v109, v16, v109
    v_cvt_pk_bf16_f32 v64, v64, v65
    v_cvt_pk_bf16_f32 v65, v66, v67
    v_cvt_pk_bf16_f32 v66, v68, v69
    v_cvt_pk_bf16_f32 v67, v70, v71
    v_cvt_pk_bf16_f32 v68, v72, v73
    v_mfma_f32_32x32x16_bf16 v[48:63], v[248:251], v[184:187], v[48:63]
    v_cvt_pk_bf16_f32 v69, v74, v75
    v_cvt_pk_bf16_f32 v70, v76, v77
    v_cvt_pk_bf16_f32 v71, v78, v79
    v_cvt_pk_bf16_f32 v72, v80, v81
    v_cvt_pk_bf16_f32 v73, v82, v83
    v_cvt_pk_bf16_f32 v74, v84, v85
    v_mfma_f32_32x32x16_bf16 v[48:63], v[252:255], v[188:191], v[48:63]
    v_cvt_pk_bf16_f32 v75, v86, v87
    v_cvt_pk_bf16_f32 v76, v88, v89
    v_cvt_pk_bf16_f32 v77, v90, v91
    v_cvt_pk_bf16_f32 v78, v92, v93
    v_cvt_pk_bf16_f32 v79, v94, v95
    s_waitcnt vmcnt(4)
    s_barrier
    s_mov_b32 m0, s64
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_add_i32 s34, s43, s34
    s_nop 0
    s_add_u32 s40, 0x100, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s43, s43, 0
    ds_read_b64_tr_b16 v[192:193], v11
    ds_read_b64_tr_b16 v[194:195], v11 offset:512
    ds_read_b64_tr_b16 v[208:209], v11 offset:64
    ds_read_b64_tr_b16 v[210:211], v11 offset:576
    ds_read_b64_tr_b16 v[196:197], v11 offset:2176
    ds_read_b64_tr_b16 v[198:199], v11 offset:2688
    ds_read_b64_tr_b16 v[212:213], v11 offset:2240
    ds_read_b64_tr_b16 v[214:215], v11 offset:2752
    ds_read_b64_tr_b16 v[200:201], v11 offset:4352
    ds_read_b64_tr_b16 v[202:203], v11 offset:4864
    ds_read_b64_tr_b16 v[216:217], v11 offset:4416
    ds_read_b64_tr_b16 v[218:219], v11 offset:4928
    ds_read_b64_tr_b16 v[204:205], v11 offset:6528
    ds_read_b64_tr_b16 v[206:207], v11 offset:7040
    ds_read_b64_tr_b16 v[220:221], v11 offset:6592
    ds_read_b64_tr_b16 v[222:223], v11 offset:7104
    ds_read_b64_tr_b16 v[224:225], v11 offset:8704
    ds_read_b64_tr_b16 v[226:227], v11 offset:9216
    ds_read_b64_tr_b16 v[240:241], v11 offset:8768
    ds_read_b64_tr_b16 v[242:243], v11 offset:9280
    ds_read_b64_tr_b16 v[228:229], v11 offset:10880
    ds_read_b64_tr_b16 v[230:231], v11 offset:11392
    ds_read_b64_tr_b16 v[244:245], v11 offset:10944
    ds_read_b64_tr_b16 v[246:247], v11 offset:11456
    ds_read_b64_tr_b16 v[232:233], v11 offset:13056
    ds_read_b64_tr_b16 v[234:235], v11 offset:13568
    ds_read_b64_tr_b16 v[248:249], v11 offset:13120
    ds_read_b64_tr_b16 v[250:251], v11 offset:13632
    ds_read_b64_tr_b16 v[236:237], v11 offset:15232
    ds_read_b64_tr_b16 v[238:239], v11 offset:15744
    ds_read_b64_tr_b16 v[252:253], v11 offset:15296
    ds_read_b64_tr_b16 v[254:255], v11 offset:15808
    s_cmp_lt_i32 s52, s54
    s_cbranch_scc1 label_0790
    s_sub_i32 s40, s7, s52
    v_sub_i32 v12, s40, v26
    v_cmp_lt_i32_e64 s[68:69], 0, v12
    v_cmp_lt_i32_e64 s[70:71], 1, v12
    v_cndmask_b32_e64 v32, v27, v32, s[68:69]
    v_cndmask_b32_e64 v33, v27, v33, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 2, v12
    v_cmp_lt_i32_e64 s[70:71], 3, v12
    v_cndmask_b32_e64 v34, v27, v34, s[68:69]
    v_cndmask_b32_e64 v35, v27, v35, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 8, v12
    v_cmp_lt_i32_e64 s[70:71], 9, v12
    v_cndmask_b32_e64 v36, v27, v36, s[68:69]
    v_cndmask_b32_e64 v37, v27, v37, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 10, v12
    v_cmp_lt_i32_e64 s[70:71], 11, v12
    v_cndmask_b32_e64 v38, v27, v38, s[68:69]
    v_cndmask_b32_e64 v39, v27, v39, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 16, v12
    v_cmp_lt_i32_e64 s[70:71], 17, v12
    v_cndmask_b32_e64 v40, v27, v40, s[68:69]
    v_cndmask_b32_e64 v41, v27, v41, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 18, v12
    v_cmp_lt_i32_e64 s[70:71], 19, v12
    v_cndmask_b32_e64 v42, v27, v42, s[68:69]
    v_cndmask_b32_e64 v43, v27, v43, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 24, v12
    v_cmp_lt_i32_e64 s[70:71], 25, v12
    v_cndmask_b32_e64 v44, v27, v44, s[68:69]
    v_cndmask_b32_e64 v45, v27, v45, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 26, v12
    v_cmp_lt_i32_e64 s[70:71], 27, v12
    v_cndmask_b32_e64 v46, v27, v46, s[68:69]
    v_cndmask_b32_e64 v47, v27, v47, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 32, v12
    v_cmp_lt_i32_e64 s[70:71], 33, v12
    v_cndmask_b32_e64 v48, v27, v48, s[68:69]
    v_cndmask_b32_e64 v49, v27, v49, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 34, v12
    v_cmp_lt_i32_e64 s[70:71], 35, v12
    v_cndmask_b32_e64 v50, v27, v50, s[68:69]
    v_cndmask_b32_e64 v51, v27, v51, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 40, v12
    v_cmp_lt_i32_e64 s[70:71], 41, v12
    v_cndmask_b32_e64 v52, v27, v52, s[68:69]
    v_cndmask_b32_e64 v53, v27, v53, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 42, v12
    v_cmp_lt_i32_e64 s[70:71], 43, v12
    v_cndmask_b32_e64 v54, v27, v54, s[68:69]
    v_cndmask_b32_e64 v55, v27, v55, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 48, v12
    v_cmp_lt_i32_e64 s[70:71], 49, v12
    v_cndmask_b32_e64 v56, v27, v56, s[68:69]
    v_cndmask_b32_e64 v57, v27, v57, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 50, v12
    v_cmp_lt_i32_e64 s[70:71], 51, v12
    v_cndmask_b32_e64 v58, v27, v58, s[68:69]
    v_cndmask_b32_e64 v59, v27, v59, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 56, v12
    v_cmp_lt_i32_e64 s[70:71], 57, v12
    v_cndmask_b32_e64 v60, v27, v60, s[68:69]
    v_cndmask_b32_e64 v61, v27, v61, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 58, v12
    v_cmp_lt_i32_e64 s[70:71], 59, v12
    v_cndmask_b32_e64 v62, v27, v62, s[68:69]
    v_cndmask_b32_e64 v63, v27, v63, s[70:71]
    s_nop 0
    s_nop 0
    s_nop 0
    s_waitcnt lgkmcnt(0)
    s_barrier
    v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[64:67], v[96:111]
    v_mov_b32_e32 v21, v24
    v_max3_f32 v21, v32, v33, v21
    v_max3_f32 v21, v34, v35, v21
    v_max3_f32 v21, v36, v37, v21
    v_max3_f32 v21, v38, v39, v21
    v_max3_f32 v21, v40, v41, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[196:199], v[68:71], v[96:111]
    v_max3_f32 v21, v42, v43, v21
    v_max3_f32 v21, v44, v45, v21
    v_max3_f32 v21, v46, v47, v21
    v_max3_f32 v21, v48, v49, v21
    v_max3_f32 v21, v50, v51, v21
    v_max3_f32 v21, v52, v53, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[200:203], v[72:75], v[96:111]
    v_max3_f32 v21, v54, v55, v21
    v_max3_f32 v21, v56, v57, v21
    v_max3_f32 v21, v58, v59, v21
    v_max3_f32 v21, v60, v61, v21
    v_max3_f32 v21, v62, v63, v21
    v_mov_b32_e32 v20, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[204:207], v[76:79], v[96:111]
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21
    v_sub_f32_e32 v16, v24, v21
    v_mfma_f32_32x32x16_bf16 v[112:127], v[208:211], v[64:67], v[112:127]
    v_mov_b32_e32 v24, v21
    v_mul_f32_e32 v23, s37, v21
    v_mul_f32_e32 v16, s37, v16
    v_exp_f32_e32 v16, v16
    v_fma_f32 v32, v32, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[212:215], v[68:71], v[112:127]
    v_fma_f32 v33, v33, s37, -v23
    v_fma_f32 v34, v34, s37, -v23
    v_fma_f32 v35, v35, s37, -v23
    v_fma_f32 v36, v36, s37, -v23
    v_fma_f32 v37, v37, s37, -v23
    v_fma_f32 v38, v38, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[216:219], v[72:75], v[112:127]
    v_fma_f32 v39, v39, s37, -v23
    v_fma_f32 v40, v40, s37, -v23
    v_fma_f32 v41, v41, s37, -v23
    v_fma_f32 v42, v42, s37, -v23
    v_fma_f32 v43, v43, s37, -v23
    v_fma_f32 v44, v44, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[76:79], v[112:127]
    v_fma_f32 v45, v45, s37, -v23
    v_fma_f32 v46, v46, s37, -v23
    v_fma_f32 v47, v47, s37, -v23
    v_fma_f32 v48, v48, s37, -v23
    v_fma_f32 v49, v49, s37, -v23
    v_fma_f32 v50, v50, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[64:67], v[128:143]
    v_fma_f32 v51, v51, s37, -v23
    v_fma_f32 v52, v52, s37, -v23
    v_fma_f32 v53, v53, s37, -v23
    v_fma_f32 v54, v54, s37, -v23
    v_fma_f32 v55, v55, s37, -v23
    v_fma_f32 v56, v56, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[68:71], v[128:143]
    v_fma_f32 v57, v57, s37, -v23
    v_fma_f32 v58, v58, s37, -v23
    v_fma_f32 v59, v59, s37, -v23
    v_fma_f32 v60, v60, s37, -v23
    v_fma_f32 v61, v61, s37, -v23
    v_fma_f32 v62, v62, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[72:75], v[128:143]
    v_fma_f32 v63, v63, s37, -v23
    v_exp_f32_e32 v32, v32
    v_exp_f32_e32 v33, v33
    v_exp_f32_e32 v34, v34
    v_mfma_f32_32x32x16_bf16 v[128:143], v[236:239], v[76:79], v[128:143]
    v_exp_f32_e32 v35, v35
    v_exp_f32_e32 v36, v36
    v_exp_f32_e32 v37, v37
    v_mfma_f32_32x32x16_bf16 v[144:159], v[240:243], v[64:67], v[144:159]
    v_exp_f32_e32 v38, v38
    v_exp_f32_e32 v39, v39
    v_exp_f32_e32 v40, v40
    v_mfma_f32_32x32x16_bf16 v[144:159], v[244:247], v[68:71], v[144:159]
    v_exp_f32_e32 v41, v41
    v_exp_f32_e32 v42, v42
    v_exp_f32_e32 v43, v43
    v_mfma_f32_32x32x16_bf16 v[144:159], v[248:251], v[72:75], v[144:159]
    v_exp_f32_e32 v44, v44
    v_exp_f32_e32 v45, v45
    v_exp_f32_e32 v46, v46
    v_mfma_f32_32x32x16_bf16 v[144:159], v[252:255], v[76:79], v[144:159]
    v_exp_f32_e32 v47, v47
    v_exp_f32_e32 v48, v48
    v_exp_f32_e32 v49, v49
    s_nop 0
    v_mov_b32_e32 v17, v16
    v_mul_f32_e32 v110, v16, v110
    v_mul_f32_e32 v111, v16, v111
    v_pk_mul_f32 v[112:113], v[16:17], v[112:113]
    v_pk_mul_f32 v[114:115], v[16:17], v[114:115]
    v_pk_mul_f32 v[116:117], v[16:17], v[116:117]
    v_pk_mul_f32 v[118:119], v[16:17], v[118:119]
    v_pk_mul_f32 v[120:121], v[16:17], v[120:121]
    v_pk_mul_f32 v[122:123], v[16:17], v[122:123]
    v_pk_mul_f32 v[124:125], v[16:17], v[124:125]
    v_pk_mul_f32 v[126:127], v[16:17], v[126:127]
    v_pk_mul_f32 v[128:129], v[16:17], v[128:129]
    v_pk_mul_f32 v[130:131], v[16:17], v[130:131]
    v_pk_mul_f32 v[132:133], v[16:17], v[132:133]
    v_pk_mul_f32 v[134:135], v[16:17], v[134:135]
    v_pk_mul_f32 v[136:137], v[16:17], v[136:137]
    v_pk_mul_f32 v[138:139], v[16:17], v[138:139]
    v_pk_mul_f32 v[140:141], v[16:17], v[140:141]
    v_pk_mul_f32 v[142:143], v[16:17], v[142:143]
    v_pk_mul_f32 v[144:145], v[16:17], v[144:145]
    v_pk_mul_f32 v[146:147], v[16:17], v[146:147]
    v_pk_mul_f32 v[148:149], v[16:17], v[148:149]
    v_pk_mul_f32 v[150:151], v[16:17], v[150:151]
    v_pk_mul_f32 v[152:153], v[16:17], v[152:153]
    v_pk_mul_f32 v[154:155], v[16:17], v[154:155]
    v_pk_mul_f32 v[156:157], v[16:17], v[156:157]
    v_pk_mul_f32 v[158:159], v[16:17], v[158:159]
    s_nop 0
    s_waitcnt vmcnt(4)
    s_barrier
    s_nop 15
    s_nop 7
    s_mov_b32 m0, s67
    buffer_load_dwordx4 v6, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    buffer_load_dwordx4 v7, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    s_add_i32 s35, s44, s35
    s_nop 0
    s_add_u32 s40, 0xc0, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s44, s44, 0
    s_nop 15
    s_nop 15
    s_nop 7
    ds_read_b128 v[192:195], v9
    ds_read_b128 v[208:211], v9 offset:512
    ds_read_b128 v[196:199], v9 offset:32
    ds_read_b128 v[212:215], v9 offset:544
    ds_read_b128 v[200:203], v9 offset:64
    ds_read_b128 v[216:219], v9 offset:576
    ds_read_b128 v[204:207], v9 offset:96
    ds_read_b128 v[220:223], v9 offset:608
    ds_read_b128 v[224:227], v9 offset:8320
    ds_read_b128 v[240:243], v9 offset:8832
    ds_read_b128 v[228:231], v9 offset:8352
    ds_read_b128 v[244:247], v9 offset:8864
    ds_read_b128 v[232:235], v9 offset:8384
    ds_read_b128 v[248:251], v9 offset:8896
    ds_read_b128 v[236:239], v9 offset:8416
    ds_read_b128 v[252:255], v9 offset:8928
    s_add_i32 s52, s52, s53
    s_addk_i32 s39, 0x40
    s_cmp_lt_i32 s39, s38
    s_cbranch_scc0 label_0D6A
    s_branch label_03CC
    s_setprio 1
    s_barrier
    s_mov_b32 m0, s67
    buffer_load_dwordx4 v6, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    buffer_load_dwordx4 v7, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    s_add_i32 s35, s44, s35
    s_nop 0
    s_add_u32 s40, 0x80, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s44, s44, 0
    ds_read_b128 v[192:195], v9
    ds_read_b128 v[208:211], v9 offset:512
    ds_read_b128 v[196:199], v9 offset:32
    ds_read_b128 v[212:215], v9 offset:544
    ds_read_b128 v[200:203], v9 offset:64
    ds_read_b128 v[216:219], v9 offset:576
    ds_read_b128 v[204:207], v9 offset:96
    ds_read_b128 v[220:223], v9 offset:608
    ds_read_b128 v[224:227], v9 offset:8320
    ds_read_b128 v[240:243], v9 offset:8832
    ds_read_b128 v[228:231], v9 offset:8352
    ds_read_b128 v[244:247], v9 offset:8864
    ds_read_b128 v[232:235], v9 offset:8384
    ds_read_b128 v[248:251], v9 offset:8896
    ds_read_b128 v[236:239], v9 offset:8416
    ds_read_b128 v[252:255], v9 offset:8928
    s_waitcnt vmcnt(4) lgkmcnt(0)
    s_barrier
    v_mfma_f32_32x32x16_bf16 v[64:79], v[192:195], v[160:163], 0
    v_exp_f32_e32 v50, v50
    v_exp_f32_e32 v51, v51
    v_exp_f32_e32 v52, v52
    v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[164:167], v[64:79]
    v_exp_f32_e32 v53, v53
    v_exp_f32_e32 v54, v54
    v_exp_f32_e32 v55, v55
    v_mfma_f32_32x32x16_bf16 v[64:79], v[200:203], v[168:171], v[64:79]
    v_exp_f32_e32 v56, v56
    v_exp_f32_e32 v57, v57
    v_exp_f32_e32 v58, v58
    v_mfma_f32_32x32x16_bf16 v[64:79], v[204:207], v[172:175], v[64:79]
    v_exp_f32_e32 v59, v59
    v_exp_f32_e32 v60, v60
    v_exp_f32_e32 v61, v61
    v_mfma_f32_32x32x16_bf16 v[64:79], v[224:227], v[176:179], v[64:79]
    v_exp_f32_e32 v62, v62
    v_exp_f32_e32 v63, v63
    v_mul_f32_e32 v18, v16, v18
    v_add_f32_e32 v19, v33, v32
    v_mfma_f32_32x32x16_bf16 v[64:79], v[228:231], v[180:183], v[64:79]
    v_add_f32_e32 v19, v34, v19
    v_add_f32_e32 v19, v35, v19
    v_add_f32_e32 v19, v36, v19
    v_add_f32_e32 v19, v37, v19
    v_add_f32_e32 v19, v38, v19
    v_add_f32_e32 v19, v39, v19
    v_mfma_f32_32x32x16_bf16 v[64:79], v[232:235], v[184:187], v[64:79]
    v_add_f32_e32 v19, v40, v19
    v_add_f32_e32 v19, v41, v19
    v_add_f32_e32 v19, v42, v19
    v_add_f32_e32 v19, v43, v19
    v_add_f32_e32 v19, v44, v19
    v_add_f32_e32 v19, v45, v19
    v_mfma_f32_32x32x16_bf16 v[64:79], v[236:239], v[188:191], v[64:79]
    v_add_f32_e32 v19, v46, v19
    v_add_f32_e32 v19, v47, v19
    v_add_f32_e32 v19, v48, v19
    v_add_f32_e32 v19, v49, v19
    v_add_f32_e32 v19, v50, v19
    v_add_f32_e32 v19, v51, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[208:211], v[160:163], 0
    v_add_f32_e32 v19, v52, v19
    v_add_f32_e32 v19, v53, v19
    v_add_f32_e32 v19, v54, v19
    v_add_f32_e32 v19, v55, v19
    v_add_f32_e32 v19, v56, v19
    v_add_f32_e32 v19, v57, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[212:215], v[164:167], v[80:95]
    v_add_f32_e32 v19, v58, v19
    v_add_f32_e32 v19, v59, v19
    v_add_f32_e32 v19, v60, v19
    v_add_f32_e32 v19, v61, v19
    v_add_f32_e32 v19, v62, v19
    v_add_f32_e32 v19, v63, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[216:219], v[168:171], v[80:95]
    v_mov_b32_e32 v20, v19
    v_mul_f32_e32 v96, v16, v96
    v_mul_f32_e32 v97, v16, v97
    v_permlane32_swap_b32_e32 v20, v19
    v_add_f32_e32 v20, v20, v19
    v_mfma_f32_32x32x16_bf16 v[80:95], v[220:223], v[172:175], v[80:95]
    v_add_f32_e32 v18, v20, v18
    v_mul_f32_e32 v98, v16, v98
    v_mul_f32_e32 v99, v16, v99
    v_mul_f32_e32 v100, v16, v100
    v_mul_f32_e32 v101, v16, v101
    v_mul_f32_e32 v102, v16, v102
    v_mfma_f32_32x32x16_bf16 v[80:95], v[240:243], v[176:179], v[80:95]
    v_mul_f32_e32 v103, v16, v103
    v_mul_f32_e32 v104, v16, v104
    v_mul_f32_e32 v105, v16, v105
    v_mul_f32_e32 v106, v16, v106
    v_mul_f32_e32 v107, v16, v107
    v_mul_f32_e32 v108, v16, v108
    v_mfma_f32_32x32x16_bf16 v[80:95], v[244:247], v[180:183], v[80:95]
    v_mul_f32_e32 v109, v16, v109
    v_cvt_pk_bf16_f32 v32, v32, v33
    v_cvt_pk_bf16_f32 v33, v34, v35
    v_cvt_pk_bf16_f32 v34, v36, v37
    v_cvt_pk_bf16_f32 v35, v38, v39
    v_cvt_pk_bf16_f32 v36, v40, v41
    v_mfma_f32_32x32x16_bf16 v[80:95], v[248:251], v[184:187], v[80:95]
    v_cvt_pk_bf16_f32 v37, v42, v43
    v_cvt_pk_bf16_f32 v38, v44, v45
    v_cvt_pk_bf16_f32 v39, v46, v47
    v_cvt_pk_bf16_f32 v40, v48, v49
    v_cvt_pk_bf16_f32 v41, v50, v51
    v_cvt_pk_bf16_f32 v42, v52, v53
    v_mfma_f32_32x32x16_bf16 v[80:95], v[252:255], v[188:191], v[80:95]
    v_cvt_pk_bf16_f32 v43, v54, v55
    v_cvt_pk_bf16_f32 v44, v56, v57
    v_cvt_pk_bf16_f32 v45, v58, v59
    v_cvt_pk_bf16_f32 v46, v60, v61
    v_cvt_pk_bf16_f32 v47, v62, v63
    s_barrier
    s_cmp_lt_i32 s52, s54
    s_cbranch_scc1 label_09DD
    s_sub_i32 s40, s7, s52
    v_sub_i32 v12, s40, v26
    v_cmp_lt_i32_e64 s[68:69], 0, v12
    v_cmp_lt_i32_e64 s[70:71], 1, v12
    v_cndmask_b32_e64 v64, v27, v64, s[68:69]
    v_cndmask_b32_e64 v65, v27, v65, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 2, v12
    v_cmp_lt_i32_e64 s[70:71], 3, v12
    v_cndmask_b32_e64 v66, v27, v66, s[68:69]
    v_cndmask_b32_e64 v67, v27, v67, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 8, v12
    v_cmp_lt_i32_e64 s[70:71], 9, v12
    v_cndmask_b32_e64 v68, v27, v68, s[68:69]
    v_cndmask_b32_e64 v69, v27, v69, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 10, v12
    v_cmp_lt_i32_e64 s[70:71], 11, v12
    v_cndmask_b32_e64 v70, v27, v70, s[68:69]
    v_cndmask_b32_e64 v71, v27, v71, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 16, v12
    v_cmp_lt_i32_e64 s[70:71], 17, v12
    v_cndmask_b32_e64 v72, v27, v72, s[68:69]
    v_cndmask_b32_e64 v73, v27, v73, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 18, v12
    v_cmp_lt_i32_e64 s[70:71], 19, v12
    v_cndmask_b32_e64 v74, v27, v74, s[68:69]
    v_cndmask_b32_e64 v75, v27, v75, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 24, v12
    v_cmp_lt_i32_e64 s[70:71], 25, v12
    v_cndmask_b32_e64 v76, v27, v76, s[68:69]
    v_cndmask_b32_e64 v77, v27, v77, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 26, v12
    v_cmp_lt_i32_e64 s[70:71], 27, v12
    v_cndmask_b32_e64 v78, v27, v78, s[68:69]
    v_cndmask_b32_e64 v79, v27, v79, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 32, v12
    v_cmp_lt_i32_e64 s[70:71], 33, v12
    v_cndmask_b32_e64 v80, v27, v80, s[68:69]
    v_cndmask_b32_e64 v81, v27, v81, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 34, v12
    v_cmp_lt_i32_e64 s[70:71], 35, v12
    v_cndmask_b32_e64 v82, v27, v82, s[68:69]
    v_cndmask_b32_e64 v83, v27, v83, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 40, v12
    v_cmp_lt_i32_e64 s[70:71], 41, v12
    v_cndmask_b32_e64 v84, v27, v84, s[68:69]
    v_cndmask_b32_e64 v85, v27, v85, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 42, v12
    v_cmp_lt_i32_e64 s[70:71], 43, v12
    v_cndmask_b32_e64 v86, v27, v86, s[68:69]
    v_cndmask_b32_e64 v87, v27, v87, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 48, v12
    v_cmp_lt_i32_e64 s[70:71], 49, v12
    v_cndmask_b32_e64 v88, v27, v88, s[68:69]
    v_cndmask_b32_e64 v89, v27, v89, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 50, v12
    v_cmp_lt_i32_e64 s[70:71], 51, v12
    v_cndmask_b32_e64 v90, v27, v90, s[68:69]
    v_cndmask_b32_e64 v91, v27, v91, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 56, v12
    v_cmp_lt_i32_e64 s[70:71], 57, v12
    v_cndmask_b32_e64 v92, v27, v92, s[68:69]
    v_cndmask_b32_e64 v93, v27, v93, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 58, v12
    v_cmp_lt_i32_e64 s[70:71], 59, v12
    v_cndmask_b32_e64 v94, v27, v94, s[68:69]
    v_cndmask_b32_e64 v95, v27, v95, s[70:71]
    s_mov_b32 m0, s65
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_add_i32 s34, s43, s34
    s_nop 0
    s_add_u32 s40, 0x100, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s43, s43, 0
    s_nop 15
    s_nop 8
    s_nop 7
    ds_read_b64_tr_b16 v[192:193], v10
    ds_read_b64_tr_b16 v[194:195], v10 offset:512
    ds_read_b64_tr_b16 v[208:209], v10 offset:64
    ds_read_b64_tr_b16 v[210:211], v10 offset:576
    ds_read_b64_tr_b16 v[196:197], v10 offset:2176
    ds_read_b64_tr_b16 v[198:199], v10 offset:2688
    ds_read_b64_tr_b16 v[212:213], v10 offset:2240
    ds_read_b64_tr_b16 v[214:215], v10 offset:2752
    ds_read_b64_tr_b16 v[200:201], v10 offset:4352
    ds_read_b64_tr_b16 v[202:203], v10 offset:4864
    ds_read_b64_tr_b16 v[216:217], v10 offset:4416
    ds_read_b64_tr_b16 v[218:219], v10 offset:4928
    ds_read_b64_tr_b16 v[204:205], v10 offset:6528
    ds_read_b64_tr_b16 v[206:207], v10 offset:7040
    ds_read_b64_tr_b16 v[220:221], v10 offset:6592
    ds_read_b64_tr_b16 v[222:223], v10 offset:7104
    ds_read_b64_tr_b16 v[224:225], v10 offset:8704
    ds_read_b64_tr_b16 v[226:227], v10 offset:9216
    ds_read_b64_tr_b16 v[240:241], v10 offset:8768
    ds_read_b64_tr_b16 v[242:243], v10 offset:9280
    ds_read_b64_tr_b16 v[228:229], v10 offset:10880
    ds_read_b64_tr_b16 v[230:231], v10 offset:11392
    ds_read_b64_tr_b16 v[244:245], v10 offset:10944
    ds_read_b64_tr_b16 v[246:247], v10 offset:11456
    ds_read_b64_tr_b16 v[232:233], v10 offset:13056
    ds_read_b64_tr_b16 v[234:235], v10 offset:13568
    ds_read_b64_tr_b16 v[248:249], v10 offset:13120
    ds_read_b64_tr_b16 v[250:251], v10 offset:13632
    ds_read_b64_tr_b16 v[236:237], v10 offset:15232
    ds_read_b64_tr_b16 v[238:239], v10 offset:15744
    ds_read_b64_tr_b16 v[252:253], v10 offset:15296
    ds_read_b64_tr_b16 v[254:255], v10 offset:15808
    s_add_i32 s52, s52, s53
    s_addk_i32 s39, 0x40
    s_cmp_lt_i32 s39, s38
    s_waitcnt vmcnt(4) lgkmcnt(0)
    s_barrier
    v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[32:35], v[96:111]
    v_mov_b32_e32 v21, v24
    v_max3_f32 v21, v64, v65, v21
    v_max3_f32 v21, v66, v67, v21
    v_max3_f32 v21, v68, v69, v21
    v_max3_f32 v21, v70, v71, v21
    v_max3_f32 v21, v72, v73, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[196:199], v[36:39], v[96:111]
    v_max3_f32 v21, v74, v75, v21
    v_max3_f32 v21, v76, v77, v21
    v_max3_f32 v21, v78, v79, v21
    v_max3_f32 v21, v80, v81, v21
    v_max3_f32 v21, v82, v83, v21
    v_max3_f32 v21, v84, v85, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[200:203], v[40:43], v[96:111]
    v_max3_f32 v21, v86, v87, v21
    v_max3_f32 v21, v88, v89, v21
    v_max3_f32 v21, v90, v91, v21
    v_max3_f32 v21, v92, v93, v21
    v_max3_f32 v21, v94, v95, v21
    v_mov_b32_e32 v20, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[204:207], v[44:47], v[96:111]
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21
    v_sub_f32_e32 v16, v24, v21
    v_mfma_f32_32x32x16_bf16 v[112:127], v[208:211], v[32:35], v[112:127]
    v_mov_b32_e32 v24, v21
    v_mul_f32_e32 v23, s37, v21
    v_mul_f32_e32 v16, s37, v16
    v_exp_f32_e32 v16, v16
    v_fma_f32 v64, v64, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[212:215], v[36:39], v[112:127]
    v_fma_f32 v65, v65, s37, -v23
    v_fma_f32 v66, v66, s37, -v23
    v_fma_f32 v67, v67, s37, -v23
    v_fma_f32 v68, v68, s37, -v23
    v_fma_f32 v69, v69, s37, -v23
    v_fma_f32 v70, v70, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[216:219], v[40:43], v[112:127]
    v_fma_f32 v71, v71, s37, -v23
    v_fma_f32 v72, v72, s37, -v23
    v_fma_f32 v73, v73, s37, -v23
    v_fma_f32 v74, v74, s37, -v23
    v_fma_f32 v75, v75, s37, -v23
    v_fma_f32 v76, v76, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[44:47], v[112:127]
    v_fma_f32 v77, v77, s37, -v23
    v_fma_f32 v78, v78, s37, -v23
    v_fma_f32 v79, v79, s37, -v23
    v_fma_f32 v80, v80, s37, -v23
    v_fma_f32 v81, v81, s37, -v23
    v_fma_f32 v82, v82, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[32:35], v[128:143]
    v_fma_f32 v83, v83, s37, -v23
    v_fma_f32 v84, v84, s37, -v23
    v_fma_f32 v85, v85, s37, -v23
    v_fma_f32 v86, v86, s37, -v23
    v_fma_f32 v87, v87, s37, -v23
    v_fma_f32 v88, v88, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[36:39], v[128:143]
    v_fma_f32 v89, v89, s37, -v23
    v_fma_f32 v90, v90, s37, -v23
    v_fma_f32 v91, v91, s37, -v23
    v_fma_f32 v92, v92, s37, -v23
    v_fma_f32 v93, v93, s37, -v23
    v_fma_f32 v94, v94, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[40:43], v[128:143]
    v_fma_f32 v95, v95, s37, -v23
    v_exp_f32_e32 v64, v64
    v_exp_f32_e32 v65, v65
    v_exp_f32_e32 v66, v66
    v_mfma_f32_32x32x16_bf16 v[128:143], v[236:239], v[44:47], v[128:143]
    v_exp_f32_e32 v67, v67
    v_exp_f32_e32 v68, v68
    v_exp_f32_e32 v69, v69
    v_mfma_f32_32x32x16_bf16 v[144:159], v[240:243], v[32:35], v[144:159]
    v_exp_f32_e32 v70, v70
    v_exp_f32_e32 v71, v71
    v_exp_f32_e32 v72, v72
    v_mfma_f32_32x32x16_bf16 v[144:159], v[244:247], v[36:39], v[144:159]
    v_exp_f32_e32 v73, v73
    v_exp_f32_e32 v74, v74
    v_exp_f32_e32 v75, v75
    v_mfma_f32_32x32x16_bf16 v[144:159], v[248:251], v[40:43], v[144:159]
    v_exp_f32_e32 v76, v76
    v_exp_f32_e32 v77, v77
    v_exp_f32_e32 v78, v78
    v_mfma_f32_32x32x16_bf16 v[144:159], v[252:255], v[44:47], v[144:159]
    v_exp_f32_e32 v79, v79
    v_exp_f32_e32 v80, v80
    v_exp_f32_e32 v81, v81
    v_nop
    v_mov_b32_e32 v17, v16
    v_mul_f32_e32 v110, v16, v110
    v_mul_f32_e32 v111, v16, v111
    v_pk_mul_f32 v[112:113], v[16:17], v[112:113]
    v_pk_mul_f32 v[114:115], v[16:17], v[114:115]
    v_pk_mul_f32 v[116:117], v[16:17], v[116:117]
    v_pk_mul_f32 v[118:119], v[16:17], v[118:119]
    v_pk_mul_f32 v[120:121], v[16:17], v[120:121]
    v_pk_mul_f32 v[122:123], v[16:17], v[122:123]
    v_pk_mul_f32 v[124:125], v[16:17], v[124:125]
    v_pk_mul_f32 v[126:127], v[16:17], v[126:127]
    v_pk_mul_f32 v[128:129], v[16:17], v[128:129]
    v_pk_mul_f32 v[130:131], v[16:17], v[130:131]
    v_pk_mul_f32 v[132:133], v[16:17], v[132:133]
    v_pk_mul_f32 v[134:135], v[16:17], v[134:135]
    v_pk_mul_f32 v[136:137], v[16:17], v[136:137]
    v_pk_mul_f32 v[138:139], v[16:17], v[138:139]
    v_pk_mul_f32 v[140:141], v[16:17], v[140:141]
    v_pk_mul_f32 v[142:143], v[16:17], v[142:143]
    v_pk_mul_f32 v[144:145], v[16:17], v[144:145]
    v_pk_mul_f32 v[146:147], v[16:17], v[146:147]
    v_pk_mul_f32 v[148:149], v[16:17], v[148:149]
    v_pk_mul_f32 v[150:151], v[16:17], v[150:151]
    v_pk_mul_f32 v[152:153], v[16:17], v[152:153]
    v_pk_mul_f32 v[154:155], v[16:17], v[154:155]
    v_pk_mul_f32 v[156:157], v[16:17], v[156:157]
    v_pk_mul_f32 v[158:159], v[16:17], v[158:159]
    s_cbranch_scc0 label_0D6A
    s_mov_b32 m0, s66
    buffer_load_dwordx4 v6, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    buffer_load_dwordx4 v7, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2200, m0
    s_add_i32 s35, s44, s35
    s_nop 0
    s_add_u32 s40, 0x80, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s44, s44, 0
    ds_read_b128 v[192:195], v8
    ds_read_b128 v[208:211], v8 offset:512
    ds_read_b128 v[196:199], v8 offset:32
    ds_read_b128 v[212:215], v8 offset:544
    ds_read_b128 v[200:203], v8 offset:64
    ds_read_b128 v[216:219], v8 offset:576
    ds_read_b128 v[204:207], v8 offset:96
    ds_read_b128 v[220:223], v8 offset:608
    ds_read_b128 v[224:227], v8 offset:8320
    ds_read_b128 v[240:243], v8 offset:8832
    ds_read_b128 v[228:231], v8 offset:8352
    ds_read_b128 v[244:247], v8 offset:8864
    ds_read_b128 v[232:235], v8 offset:8384
    ds_read_b128 v[248:251], v8 offset:8896
    ds_read_b128 v[236:239], v8 offset:8416
    ds_read_b128 v[252:255], v8 offset:8928
    s_waitcnt vmcnt(4) lgkmcnt(0)
    s_barrier
    v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[160:163], 0
    v_exp_f32_e32 v82, v82
    v_exp_f32_e32 v83, v83
    v_exp_f32_e32 v84, v84
    v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[164:167], v[32:47]
    v_exp_f32_e32 v85, v85
    v_exp_f32_e32 v86, v86
    v_exp_f32_e32 v87, v87
    v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[168:171], v[32:47]
    v_exp_f32_e32 v88, v88
    v_exp_f32_e32 v89, v89
    v_exp_f32_e32 v90, v90
    v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[172:175], v[32:47]
    v_exp_f32_e32 v91, v91
    v_exp_f32_e32 v92, v92
    v_exp_f32_e32 v93, v93
    v_mfma_f32_32x32x16_bf16 v[32:47], v[224:227], v[176:179], v[32:47]
    v_exp_f32_e32 v94, v94
    v_exp_f32_e32 v95, v95
    v_mul_f32_e32 v18, v16, v18
    v_add_f32_e32 v19, v65, v64
    v_mfma_f32_32x32x16_bf16 v[32:47], v[228:231], v[180:183], v[32:47]
    v_add_f32_e32 v19, v66, v19
    v_add_f32_e32 v19, v67, v19
    v_add_f32_e32 v19, v68, v19
    v_add_f32_e32 v19, v69, v19
    v_add_f32_e32 v19, v70, v19
    v_add_f32_e32 v19, v71, v19
    v_mfma_f32_32x32x16_bf16 v[32:47], v[232:235], v[184:187], v[32:47]
    v_add_f32_e32 v19, v72, v19
    v_add_f32_e32 v19, v73, v19
    v_add_f32_e32 v19, v74, v19
    v_add_f32_e32 v19, v75, v19
    v_add_f32_e32 v19, v76, v19
    v_add_f32_e32 v19, v77, v19
    v_mfma_f32_32x32x16_bf16 v[32:47], v[236:239], v[188:191], v[32:47]
    v_add_f32_e32 v19, v78, v19
    v_add_f32_e32 v19, v79, v19
    v_add_f32_e32 v19, v80, v19
    v_add_f32_e32 v19, v81, v19
    v_add_f32_e32 v19, v82, v19
    v_add_f32_e32 v19, v83, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[208:211], v[160:163], 0
    v_add_f32_e32 v19, v84, v19
    v_add_f32_e32 v19, v85, v19
    v_add_f32_e32 v19, v86, v19
    v_add_f32_e32 v19, v87, v19
    v_add_f32_e32 v19, v88, v19
    v_add_f32_e32 v19, v89, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[164:167], v[48:63]
    v_add_f32_e32 v19, v90, v19
    v_add_f32_e32 v19, v91, v19
    v_add_f32_e32 v19, v92, v19
    v_add_f32_e32 v19, v93, v19
    v_add_f32_e32 v19, v94, v19
    v_add_f32_e32 v19, v95, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[216:219], v[168:171], v[48:63]
    v_mov_b32_e32 v20, v19
    v_mul_f32_e32 v96, v16, v96
    v_mul_f32_e32 v97, v16, v97
    v_permlane32_swap_b32_e32 v20, v19
    v_add_f32_e32 v20, v20, v19
    v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[172:175], v[48:63]
    v_add_f32_e32 v18, v20, v18
    v_mul_f32_e32 v98, v16, v98
    v_mul_f32_e32 v99, v16, v99
    v_mul_f32_e32 v100, v16, v100
    v_mul_f32_e32 v101, v16, v101
    v_mul_f32_e32 v102, v16, v102
    v_mfma_f32_32x32x16_bf16 v[48:63], v[240:243], v[176:179], v[48:63]
    v_mul_f32_e32 v103, v16, v103
    v_mul_f32_e32 v104, v16, v104
    v_mul_f32_e32 v105, v16, v105
    v_mul_f32_e32 v106, v16, v106
    v_mul_f32_e32 v107, v16, v107
    v_mul_f32_e32 v108, v16, v108
    v_mfma_f32_32x32x16_bf16 v[48:63], v[244:247], v[180:183], v[48:63]
    v_mul_f32_e32 v109, v16, v109
    v_cvt_pk_bf16_f32 v64, v64, v65
    v_cvt_pk_bf16_f32 v65, v66, v67
    v_cvt_pk_bf16_f32 v66, v68, v69
    v_cvt_pk_bf16_f32 v67, v70, v71
    v_cvt_pk_bf16_f32 v68, v72, v73
    v_mfma_f32_32x32x16_bf16 v[48:63], v[248:251], v[184:187], v[48:63]
    v_cvt_pk_bf16_f32 v69, v74, v75
    v_cvt_pk_bf16_f32 v70, v76, v77
    v_cvt_pk_bf16_f32 v71, v78, v79
    v_cvt_pk_bf16_f32 v72, v80, v81
    v_cvt_pk_bf16_f32 v73, v82, v83
    v_cvt_pk_bf16_f32 v74, v84, v85
    v_mfma_f32_32x32x16_bf16 v[48:63], v[252:255], v[188:191], v[48:63]
    v_cvt_pk_bf16_f32 v75, v86, v87
    v_cvt_pk_bf16_f32 v76, v88, v89
    v_cvt_pk_bf16_f32 v77, v90, v91
    v_cvt_pk_bf16_f32 v78, v92, v93
    v_cvt_pk_bf16_f32 v79, v94, v95
    s_barrier
    s_cmp_lt_i32 s52, s54
    s_cbranch_scc1 label_0C3F
    s_sub_i32 s40, s7, s52
    v_sub_i32 v12, s40, v26
    v_cmp_lt_i32_e64 s[68:69], 0, v12
    v_cmp_lt_i32_e64 s[70:71], 1, v12
    v_cndmask_b32_e64 v32, v27, v32, s[68:69]
    v_cndmask_b32_e64 v33, v27, v33, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 2, v12
    v_cmp_lt_i32_e64 s[70:71], 3, v12
    v_cndmask_b32_e64 v34, v27, v34, s[68:69]
    v_cndmask_b32_e64 v35, v27, v35, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 8, v12
    v_cmp_lt_i32_e64 s[70:71], 9, v12
    v_cndmask_b32_e64 v36, v27, v36, s[68:69]
    v_cndmask_b32_e64 v37, v27, v37, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 10, v12
    v_cmp_lt_i32_e64 s[70:71], 11, v12
    v_cndmask_b32_e64 v38, v27, v38, s[68:69]
    v_cndmask_b32_e64 v39, v27, v39, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 16, v12
    v_cmp_lt_i32_e64 s[70:71], 17, v12
    v_cndmask_b32_e64 v40, v27, v40, s[68:69]
    v_cndmask_b32_e64 v41, v27, v41, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 18, v12
    v_cmp_lt_i32_e64 s[70:71], 19, v12
    v_cndmask_b32_e64 v42, v27, v42, s[68:69]
    v_cndmask_b32_e64 v43, v27, v43, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 24, v12
    v_cmp_lt_i32_e64 s[70:71], 25, v12
    v_cndmask_b32_e64 v44, v27, v44, s[68:69]
    v_cndmask_b32_e64 v45, v27, v45, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 26, v12
    v_cmp_lt_i32_e64 s[70:71], 27, v12
    v_cndmask_b32_e64 v46, v27, v46, s[68:69]
    v_cndmask_b32_e64 v47, v27, v47, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 32, v12
    v_cmp_lt_i32_e64 s[70:71], 33, v12
    v_cndmask_b32_e64 v48, v27, v48, s[68:69]
    v_cndmask_b32_e64 v49, v27, v49, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 34, v12
    v_cmp_lt_i32_e64 s[70:71], 35, v12
    v_cndmask_b32_e64 v50, v27, v50, s[68:69]
    v_cndmask_b32_e64 v51, v27, v51, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 40, v12
    v_cmp_lt_i32_e64 s[70:71], 41, v12
    v_cndmask_b32_e64 v52, v27, v52, s[68:69]
    v_cndmask_b32_e64 v53, v27, v53, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 42, v12
    v_cmp_lt_i32_e64 s[70:71], 43, v12
    v_cndmask_b32_e64 v54, v27, v54, s[68:69]
    v_cndmask_b32_e64 v55, v27, v55, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 48, v12
    v_cmp_lt_i32_e64 s[70:71], 49, v12
    v_cndmask_b32_e64 v56, v27, v56, s[68:69]
    v_cndmask_b32_e64 v57, v27, v57, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 50, v12
    v_cmp_lt_i32_e64 s[70:71], 51, v12
    v_cndmask_b32_e64 v58, v27, v58, s[68:69]
    v_cndmask_b32_e64 v59, v27, v59, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 56, v12
    v_cmp_lt_i32_e64 s[70:71], 57, v12
    v_cndmask_b32_e64 v60, v27, v60, s[68:69]
    v_cndmask_b32_e64 v61, v27, v61, s[70:71]
    v_cmp_lt_i32_e64 s[68:69], 58, v12
    v_cmp_lt_i32_e64 s[70:71], 59, v12
    v_cndmask_b32_e64 v62, v27, v62, s[68:69]
    v_cndmask_b32_e64 v63, v27, v63, s[70:71]
    s_mov_b32 m0, s64
    buffer_load_dwordx4 v4, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    buffer_load_dwordx4 v5, s[12:15], s34 offen lds
    s_add_u32 m0, 0x2080, m0
    s_add_i32 s34, s43, s34
    s_nop 0
    s_add_u32 s40, 0x100, s39
    s_nop 0
    s_cmp_lt_u32 s40, s38
    s_cselect_b32 s43, s43, 0
    s_nop 15
    s_nop 8
    s_nop 7
    ds_read_b64_tr_b16 v[192:193], v11
    ds_read_b64_tr_b16 v[194:195], v11 offset:512
    ds_read_b64_tr_b16 v[208:209], v11 offset:64
    ds_read_b64_tr_b16 v[210:211], v11 offset:576
    ds_read_b64_tr_b16 v[196:197], v11 offset:2176
    ds_read_b64_tr_b16 v[198:199], v11 offset:2688
    ds_read_b64_tr_b16 v[212:213], v11 offset:2240
    ds_read_b64_tr_b16 v[214:215], v11 offset:2752
    ds_read_b64_tr_b16 v[200:201], v11 offset:4352
    ds_read_b64_tr_b16 v[202:203], v11 offset:4864
    ds_read_b64_tr_b16 v[216:217], v11 offset:4416
    ds_read_b64_tr_b16 v[218:219], v11 offset:4928
    ds_read_b64_tr_b16 v[204:205], v11 offset:6528
    ds_read_b64_tr_b16 v[206:207], v11 offset:7040
    ds_read_b64_tr_b16 v[220:221], v11 offset:6592
    ds_read_b64_tr_b16 v[222:223], v11 offset:7104
    ds_read_b64_tr_b16 v[224:225], v11 offset:8704
    ds_read_b64_tr_b16 v[226:227], v11 offset:9216
    ds_read_b64_tr_b16 v[240:241], v11 offset:8768
    ds_read_b64_tr_b16 v[242:243], v11 offset:9280
    ds_read_b64_tr_b16 v[228:229], v11 offset:10880
    ds_read_b64_tr_b16 v[230:231], v11 offset:11392
    ds_read_b64_tr_b16 v[244:245], v11 offset:10944
    ds_read_b64_tr_b16 v[246:247], v11 offset:11456
    ds_read_b64_tr_b16 v[232:233], v11 offset:13056
    ds_read_b64_tr_b16 v[234:235], v11 offset:13568
    ds_read_b64_tr_b16 v[248:249], v11 offset:13120
    ds_read_b64_tr_b16 v[250:251], v11 offset:13632
    ds_read_b64_tr_b16 v[236:237], v11 offset:15232
    ds_read_b64_tr_b16 v[238:239], v11 offset:15744
    ds_read_b64_tr_b16 v[252:253], v11 offset:15296
    ds_read_b64_tr_b16 v[254:255], v11 offset:15808
    s_add_i32 s52, s52, s53
    s_addk_i32 s39, 0x40
    s_cmp_lt_i32 s39, s38
    s_waitcnt vmcnt(4) lgkmcnt(0)
    s_barrier
    v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[64:67], v[96:111]
    v_mov_b32_e32 v21, v24
    v_max3_f32 v21, v32, v33, v21
    v_max3_f32 v21, v34, v35, v21
    v_max3_f32 v21, v36, v37, v21
    v_max3_f32 v21, v38, v39, v21
    v_max3_f32 v21, v40, v41, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[196:199], v[68:71], v[96:111]
    v_max3_f32 v21, v42, v43, v21
    v_max3_f32 v21, v44, v45, v21
    v_max3_f32 v21, v46, v47, v21
    v_max3_f32 v21, v48, v49, v21
    v_max3_f32 v21, v50, v51, v21
    v_max3_f32 v21, v52, v53, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[200:203], v[72:75], v[96:111]
    v_max3_f32 v21, v54, v55, v21
    v_max3_f32 v21, v56, v57, v21
    v_max3_f32 v21, v58, v59, v21
    v_max3_f32 v21, v60, v61, v21
    v_max3_f32 v21, v62, v63, v21
    v_mov_b32_e32 v20, v21
    v_mfma_f32_32x32x16_bf16 v[96:111], v[204:207], v[76:79], v[96:111]
    v_nop
    v_nop
    v_permlane32_swap_b32_e32 v20, v21
    v_max_f32_e32 v21, v20, v21
    v_sub_f32_e32 v16, v24, v21
    v_mfma_f32_32x32x16_bf16 v[112:127], v[208:211], v[64:67], v[112:127]
    v_mov_b32_e32 v24, v21
    v_mul_f32_e32 v23, s37, v21
    v_mul_f32_e32 v16, s37, v16
    v_exp_f32_e32 v16, v16
    v_fma_f32 v32, v32, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[212:215], v[68:71], v[112:127]
    v_fma_f32 v33, v33, s37, -v23
    v_fma_f32 v34, v34, s37, -v23
    v_fma_f32 v35, v35, s37, -v23
    v_fma_f32 v36, v36, s37, -v23
    v_fma_f32 v37, v37, s37, -v23
    v_fma_f32 v38, v38, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[216:219], v[72:75], v[112:127]
    v_fma_f32 v39, v39, s37, -v23
    v_fma_f32 v40, v40, s37, -v23
    v_fma_f32 v41, v41, s37, -v23
    v_fma_f32 v42, v42, s37, -v23
    v_fma_f32 v43, v43, s37, -v23
    v_fma_f32 v44, v44, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[76:79], v[112:127]
    v_fma_f32 v45, v45, s37, -v23
    v_fma_f32 v46, v46, s37, -v23
    v_fma_f32 v47, v47, s37, -v23
    v_fma_f32 v48, v48, s37, -v23
    v_fma_f32 v49, v49, s37, -v23
    v_fma_f32 v50, v50, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[64:67], v[128:143]
    v_fma_f32 v51, v51, s37, -v23
    v_fma_f32 v52, v52, s37, -v23
    v_fma_f32 v53, v53, s37, -v23
    v_fma_f32 v54, v54, s37, -v23
    v_fma_f32 v55, v55, s37, -v23
    v_fma_f32 v56, v56, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[68:71], v[128:143]
    v_fma_f32 v57, v57, s37, -v23
    v_fma_f32 v58, v58, s37, -v23
    v_fma_f32 v59, v59, s37, -v23
    v_fma_f32 v60, v60, s37, -v23
    v_fma_f32 v61, v61, s37, -v23
    v_fma_f32 v62, v62, s37, -v23
    v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[72:75], v[128:143]
    v_fma_f32 v63, v63, s37, -v23
    v_exp_f32_e32 v32, v32
    v_exp_f32_e32 v33, v33
    v_exp_f32_e32 v34, v34
    v_mfma_f32_32x32x16_bf16 v[128:143], v[236:239], v[76:79], v[128:143]
    v_exp_f32_e32 v35, v35
    v_exp_f32_e32 v36, v36
    v_exp_f32_e32 v37, v37
    v_mfma_f32_32x32x16_bf16 v[144:159], v[240:243], v[64:67], v[144:159]
    v_exp_f32_e32 v38, v38
    v_exp_f32_e32 v39, v39
    v_exp_f32_e32 v40, v40
    v_mfma_f32_32x32x16_bf16 v[144:159], v[244:247], v[68:71], v[144:159]
    v_exp_f32_e32 v41, v41
    v_exp_f32_e32 v42, v42
    v_exp_f32_e32 v43, v43
    v_mfma_f32_32x32x16_bf16 v[144:159], v[248:251], v[72:75], v[144:159]
    v_exp_f32_e32 v44, v44
    v_exp_f32_e32 v45, v45
    v_exp_f32_e32 v46, v46
    v_mfma_f32_32x32x16_bf16 v[144:159], v[252:255], v[76:79], v[144:159]
    v_exp_f32_e32 v47, v47
    v_exp_f32_e32 v48, v48
    v_exp_f32_e32 v49, v49
    v_nop
    v_mov_b32_e32 v17, v16
    v_mul_f32_e32 v110, v16, v110
    v_mul_f32_e32 v111, v16, v111
    v_pk_mul_f32 v[112:113], v[16:17], v[112:113]
    v_pk_mul_f32 v[114:115], v[16:17], v[114:115]
    v_pk_mul_f32 v[116:117], v[16:17], v[116:117]
    v_pk_mul_f32 v[118:119], v[16:17], v[118:119]
    v_pk_mul_f32 v[120:121], v[16:17], v[120:121]
    v_pk_mul_f32 v[122:123], v[16:17], v[122:123]
    v_pk_mul_f32 v[124:125], v[16:17], v[124:125]
    v_pk_mul_f32 v[126:127], v[16:17], v[126:127]
    v_pk_mul_f32 v[128:129], v[16:17], v[128:129]
    v_pk_mul_f32 v[130:131], v[16:17], v[130:131]
    v_pk_mul_f32 v[132:133], v[16:17], v[132:133]
    v_pk_mul_f32 v[134:135], v[16:17], v[134:135]
    v_pk_mul_f32 v[136:137], v[16:17], v[136:137]
    v_pk_mul_f32 v[138:139], v[16:17], v[138:139]
    v_pk_mul_f32 v[140:141], v[16:17], v[140:141]
    v_pk_mul_f32 v[142:143], v[16:17], v[142:143]
    v_pk_mul_f32 v[144:145], v[16:17], v[144:145]
    v_pk_mul_f32 v[146:147], v[16:17], v[146:147]
    v_pk_mul_f32 v[148:149], v[16:17], v[148:149]
    v_pk_mul_f32 v[150:151], v[16:17], v[150:151]
    v_pk_mul_f32 v[152:153], v[16:17], v[152:153]
    v_pk_mul_f32 v[154:155], v[16:17], v[154:155]
    v_pk_mul_f32 v[156:157], v[16:17], v[156:157]
    v_pk_mul_f32 v[158:159], v[16:17], v[158:159]
    s_cbranch_scc0 label_0D6A
    s_branch label_08A5
    s_add_u32 s40, s38, 63
    s_lshr_b32 s40, s40, 6
    s_and_b32 s40, 1, s40
    s_cmp_lt_i32 s40, 1
    s_cbranch_scc0 label_0E33
    s_waitcnt vmcnt(2)
    s_barrier
    ds_read_b64_tr_b16 v[192:193], v11
    ds_read_b64_tr_b16 v[194:195], v11 offset:512
    ds_read_b64_tr_b16 v[208:209], v11 offset:64
    ds_read_b64_tr_b16 v[210:211], v11 offset:576
    ds_read_b64_tr_b16 v[196:197], v11 offset:2176
    ds_read_b64_tr_b16 v[198:199], v11 offset:2688
    ds_read_b64_tr_b16 v[212:213], v11 offset:2240
    ds_read_b64_tr_b16 v[214:215], v11 offset:2752
    ds_read_b64_tr_b16 v[200:201], v11 offset:4352
    ds_read_b64_tr_b16 v[202:203], v11 offset:4864
    ds_read_b64_tr_b16 v[216:217], v11 offset:4416
    ds_read_b64_tr_b16 v[218:219], v11 offset:4928
    ds_read_b64_tr_b16 v[204:205], v11 offset:6528
    ds_read_b64_tr_b16 v[206:207], v11 offset:7040
    ds_read_b64_tr_b16 v[220:221], v11 offset:6592
    ds_read_b64_tr_b16 v[222:223], v11 offset:7104
    ds_read_b64_tr_b16 v[224:225], v11 offset:8704
    ds_read_b64_tr_b16 v[226:227], v11 offset:9216
    ds_read_b64_tr_b16 v[240:241], v11 offset:8768
    ds_read_b64_tr_b16 v[242:243], v11 offset:9280
    ds_read_b64_tr_b16 v[228:229], v11 offset:10880
    ds_read_b64_tr_b16 v[230:231], v11 offset:11392
    ds_read_b64_tr_b16 v[244:245], v11 offset:10944
    ds_read_b64_tr_b16 v[246:247], v11 offset:11456
    ds_read_b64_tr_b16 v[232:233], v11 offset:13056
    ds_read_b64_tr_b16 v[234:235], v11 offset:13568
    ds_read_b64_tr_b16 v[248:249], v11 offset:13120
    ds_read_b64_tr_b16 v[250:251], v11 offset:13632
    ds_read_b64_tr_b16 v[236:237], v11 offset:15232
    ds_read_b64_tr_b16 v[238:239], v11 offset:15744
    ds_read_b64_tr_b16 v[252:253], v11 offset:15296
    ds_read_b64_tr_b16 v[254:255], v11 offset:15808
    v_exp_f32_e32 v82, v82
    v_exp_f32_e32 v83, v83
    v_exp_f32_e32 v84, v84
    v_exp_f32_e32 v85, v85
    v_exp_f32_e32 v86, v86
    v_exp_f32_e32 v87, v87
    v_exp_f32_e32 v88, v88
    v_exp_f32_e32 v89, v89
    v_exp_f32_e32 v90, v90
    v_exp_f32_e32 v91, v91
    v_exp_f32_e32 v92, v92
    v_exp_f32_e32 v93, v93
    v_exp_f32_e32 v94, v94
    v_exp_f32_e32 v95, v95
    v_mul_f32_e32 v18, v16, v18
    v_add_f32_e32 v19, v65, v64
    v_add_f32_e32 v19, v66, v19
    v_add_f32_e32 v19, v67, v19
    v_add_f32_e32 v19, v68, v19
    v_add_f32_e32 v19, v69, v19
    v_add_f32_e32 v19, v70, v19
    v_add_f32_e32 v19, v71, v19
    v_add_f32_e32 v19, v72, v19
    v_add_f32_e32 v19, v73, v19
    v_add_f32_e32 v19, v74, v19
    v_add_f32_e32 v19, v75, v19
    v_add_f32_e32 v19, v76, v19
    v_add_f32_e32 v19, v77, v19
    v_add_f32_e32 v19, v78, v19
    v_add_f32_e32 v19, v79, v19
    v_add_f32_e32 v19, v80, v19
    v_add_f32_e32 v19, v81, v19
    v_add_f32_e32 v19, v82, v19
    v_add_f32_e32 v19, v83, v19
    v_add_f32_e32 v19, v84, v19
    v_add_f32_e32 v19, v85, v19
    v_add_f32_e32 v19, v86, v19
    v_add_f32_e32 v19, v87, v19
    v_add_f32_e32 v19, v88, v19
    v_add_f32_e32 v19, v89, v19
    v_add_f32_e32 v19, v90, v19
    v_add_f32_e32 v19, v91, v19
    v_add_f32_e32 v19, v92, v19
    v_add_f32_e32 v19, v93, v19
    v_add_f32_e32 v19, v94, v19
    v_add_f32_e32 v19, v95, v19
    v_mov_b32_e32 v20, v19
    v_mul_f32_e32 v96, v16, v96
    v_mul_f32_e32 v97, v16, v97
    v_permlane32_swap_b32_e32 v20, v19
    v_add_f32_e32 v20, v20, v19
    v_add_f32_e32 v18, v20, v18
    v_mul_f32_e32 v98, v16, v98
    v_mul_f32_e32 v99, v16, v99
    v_mul_f32_e32 v100, v16, v100
    v_mul_f32_e32 v101, v16, v101
    v_mul_f32_e32 v102, v16, v102
    v_mul_f32_e32 v103, v16, v103
    v_mul_f32_e32 v104, v16, v104
    v_mul_f32_e32 v105, v16, v105
    v_mul_f32_e32 v106, v16, v106
    v_mul_f32_e32 v107, v16, v107
    v_mul_f32_e32 v108, v16, v108
    v_mul_f32_e32 v109, v16, v109
    v_cvt_pk_bf16_f32 v64, v64, v65
    v_cvt_pk_bf16_f32 v65, v66, v67
    v_cvt_pk_bf16_f32 v66, v68, v69
    v_cvt_pk_bf16_f32 v67, v70, v71
    v_cvt_pk_bf16_f32 v68, v72, v73
    v_cvt_pk_bf16_f32 v69, v74, v75
    v_cvt_pk_bf16_f32 v70, v76, v77
    v_cvt_pk_bf16_f32 v71, v78, v79
    v_cvt_pk_bf16_f32 v72, v80, v81
    v_cvt_pk_bf16_f32 v73, v82, v83
    v_cvt_pk_bf16_f32 v74, v84, v85
    v_cvt_pk_bf16_f32 v75, v86, v87
    v_cvt_pk_bf16_f32 v76, v88, v89
    v_cvt_pk_bf16_f32 v77, v90, v91
    v_cvt_pk_bf16_f32 v78, v92, v93
    v_cvt_pk_bf16_f32 v79, v94, v95
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[64:67], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[96:111], v[196:199], v[68:71], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[96:111], v[200:203], v[72:75], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[96:111], v[204:207], v[76:79], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[208:211], v[64:67], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[212:215], v[68:71], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[216:219], v[72:75], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[76:79], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[64:67], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[68:71], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[72:75], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[236:239], v[76:79], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[240:243], v[64:67], v[144:159]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[244:247], v[68:71], v[144:159]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[248:251], v[72:75], v[144:159]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[252:255], v[76:79], v[144:159]
    s_branch label_0EF6
    s_waitcnt vmcnt(2)
    s_barrier
    ds_read_b64_tr_b16 v[192:193], v10
    ds_read_b64_tr_b16 v[194:195], v10 offset:512
    ds_read_b64_tr_b16 v[208:209], v10 offset:64
    ds_read_b64_tr_b16 v[210:211], v10 offset:576
    ds_read_b64_tr_b16 v[196:197], v10 offset:2176
    ds_read_b64_tr_b16 v[198:199], v10 offset:2688
    ds_read_b64_tr_b16 v[212:213], v10 offset:2240
    ds_read_b64_tr_b16 v[214:215], v10 offset:2752
    ds_read_b64_tr_b16 v[200:201], v10 offset:4352
    ds_read_b64_tr_b16 v[202:203], v10 offset:4864
    ds_read_b64_tr_b16 v[216:217], v10 offset:4416
    ds_read_b64_tr_b16 v[218:219], v10 offset:4928
    ds_read_b64_tr_b16 v[204:205], v10 offset:6528
    ds_read_b64_tr_b16 v[206:207], v10 offset:7040
    ds_read_b64_tr_b16 v[220:221], v10 offset:6592
    ds_read_b64_tr_b16 v[222:223], v10 offset:7104
    ds_read_b64_tr_b16 v[224:225], v10 offset:8704
    ds_read_b64_tr_b16 v[226:227], v10 offset:9216
    ds_read_b64_tr_b16 v[240:241], v10 offset:8768
    ds_read_b64_tr_b16 v[242:243], v10 offset:9280
    ds_read_b64_tr_b16 v[228:229], v10 offset:10880
    ds_read_b64_tr_b16 v[230:231], v10 offset:11392
    ds_read_b64_tr_b16 v[244:245], v10 offset:10944
    ds_read_b64_tr_b16 v[246:247], v10 offset:11456
    ds_read_b64_tr_b16 v[232:233], v10 offset:13056
    ds_read_b64_tr_b16 v[234:235], v10 offset:13568
    ds_read_b64_tr_b16 v[248:249], v10 offset:13120
    ds_read_b64_tr_b16 v[250:251], v10 offset:13632
    ds_read_b64_tr_b16 v[236:237], v10 offset:15232
    ds_read_b64_tr_b16 v[238:239], v10 offset:15744
    ds_read_b64_tr_b16 v[252:253], v10 offset:15296
    ds_read_b64_tr_b16 v[254:255], v10 offset:15808
    v_exp_f32_e32 v50, v50
    v_exp_f32_e32 v51, v51
    v_exp_f32_e32 v52, v52
    v_exp_f32_e32 v53, v53
    v_exp_f32_e32 v54, v54
    v_exp_f32_e32 v55, v55
    v_exp_f32_e32 v56, v56
    v_exp_f32_e32 v57, v57
    v_exp_f32_e32 v58, v58
    v_exp_f32_e32 v59, v59
    v_exp_f32_e32 v60, v60
    v_exp_f32_e32 v61, v61
    v_exp_f32_e32 v62, v62
    v_exp_f32_e32 v63, v63
    v_mul_f32_e32 v18, v16, v18
    v_add_f32_e32 v19, v33, v32
    v_add_f32_e32 v19, v34, v19
    v_add_f32_e32 v19, v35, v19
    v_add_f32_e32 v19, v36, v19
    v_add_f32_e32 v19, v37, v19
    v_add_f32_e32 v19, v38, v19
    v_add_f32_e32 v19, v39, v19
    v_add_f32_e32 v19, v40, v19
    v_add_f32_e32 v19, v41, v19
    v_add_f32_e32 v19, v42, v19
    v_add_f32_e32 v19, v43, v19
    v_add_f32_e32 v19, v44, v19
    v_add_f32_e32 v19, v45, v19
    v_add_f32_e32 v19, v46, v19
    v_add_f32_e32 v19, v47, v19
    v_add_f32_e32 v19, v48, v19
    v_add_f32_e32 v19, v49, v19
    v_add_f32_e32 v19, v50, v19
    v_add_f32_e32 v19, v51, v19
    v_add_f32_e32 v19, v52, v19
    v_add_f32_e32 v19, v53, v19
    v_add_f32_e32 v19, v54, v19
    v_add_f32_e32 v19, v55, v19
    v_add_f32_e32 v19, v56, v19
    v_add_f32_e32 v19, v57, v19
    v_add_f32_e32 v19, v58, v19
    v_add_f32_e32 v19, v59, v19
    v_add_f32_e32 v19, v60, v19
    v_add_f32_e32 v19, v61, v19
    v_add_f32_e32 v19, v62, v19
    v_add_f32_e32 v19, v63, v19
    v_mov_b32_e32 v20, v19
    v_mul_f32_e32 v96, v16, v96
    v_mul_f32_e32 v97, v16, v97
    v_permlane32_swap_b32_e32 v20, v19
    v_add_f32_e32 v20, v20, v19
    v_add_f32_e32 v18, v20, v18
    v_mul_f32_e32 v98, v16, v98
    v_mul_f32_e32 v99, v16, v99
    v_mul_f32_e32 v100, v16, v100
    v_mul_f32_e32 v101, v16, v101
    v_mul_f32_e32 v102, v16, v102
    v_mul_f32_e32 v103, v16, v103
    v_mul_f32_e32 v104, v16, v104
    v_mul_f32_e32 v105, v16, v105
    v_mul_f32_e32 v106, v16, v106
    v_mul_f32_e32 v107, v16, v107
    v_mul_f32_e32 v108, v16, v108
    v_mul_f32_e32 v109, v16, v109
    v_cvt_pk_bf16_f32 v32, v32, v33
    v_cvt_pk_bf16_f32 v33, v34, v35
    v_cvt_pk_bf16_f32 v34, v36, v37
    v_cvt_pk_bf16_f32 v35, v38, v39
    v_cvt_pk_bf16_f32 v36, v40, v41
    v_cvt_pk_bf16_f32 v37, v42, v43
    v_cvt_pk_bf16_f32 v38, v44, v45
    v_cvt_pk_bf16_f32 v39, v46, v47
    v_cvt_pk_bf16_f32 v40, v48, v49
    v_cvt_pk_bf16_f32 v41, v50, v51
    v_cvt_pk_bf16_f32 v42, v52, v53
    v_cvt_pk_bf16_f32 v43, v54, v55
    v_cvt_pk_bf16_f32 v44, v56, v57
    v_cvt_pk_bf16_f32 v45, v58, v59
    v_cvt_pk_bf16_f32 v46, v60, v61
    v_cvt_pk_bf16_f32 v47, v62, v63
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[32:35], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[96:111], v[196:199], v[36:39], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[96:111], v[200:203], v[40:43], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[96:111], v[204:207], v[44:47], v[96:111]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[208:211], v[32:35], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[212:215], v[36:39], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[216:219], v[40:43], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[112:127], v[220:223], v[44:47], v[112:127]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[224:227], v[32:35], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[228:231], v[36:39], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[232:235], v[40:43], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[128:143], v[236:239], v[44:47], v[128:143]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[240:243], v[32:35], v[144:159]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[244:247], v[36:39], v[144:159]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[248:251], v[40:43], v[144:159]
    v_mfma_f32_32x32x16_bf16 v[144:159], v[252:255], v[44:47], v[144:159]
    v_cvt_f32_u32_e32 v12, s50
    s_sub_i32 s40, 0, s50
    v_rcp_iflag_f32_e32 v12, v12
    s_nop 0
    v_mul_f32_e32 v12, 0x4f7ffffe, v12
    v_cvt_u32_f32_e32 v12, v12
    v_mul_lo_u32 v13, s40, v12
    v_mul_hi_u32 v13, v12, v13
    v_add_u32_e32 v12, v12, v13
    v_mul_hi_u32 v12, s59, v12
    v_mul_lo_u32 v13, v12, s50
    v_sub_u32_e32 v15, s59, v13
    v_add_u32_e32 v14, 1, v12
    v_cmp_le_u32_e32 vcc, s50, v15
    v_subrev_u32_e32 v13, s50, v15
    s_nop 0
    v_cndmask_b32_e32 v12, v12, v14, vcc
    v_cndmask_b32_e32 v15, v15, v13, vcc
    v_add_u32_e32 v13, 1, v12
    v_cmp_le_u32_e32 vcc, s50, v15
    s_nop 1
    v_cndmask_b32_e32 v15, v12, v13, vcc
    s_nop 3
    v_readfirstlane_b32 s59, v15
    s_nop 3
    s_mul_i32 s59, s59, s79
    v_lshrrev_b32_e32 v12, 4, v0
    v_and_b32_e32 v13, 1, v12
    v_mul_i32_i24_e32 v13, 32, v13
    v_lshrrev_b32_e32 v14, 1, v12
    v_mul_i32_i24_e32 v14, 16, v14
    v_add_u32_e32 v13, v14, v13
    v_and_b32_e32 v12, 15, v0
    v_mul_i32_i24_e32 v12, s79, v12
    v_add_u32_e32 v12, v13, v12
    s_mul_i32 s40, s5, s79
    s_mul_i32 s40, s40, 32
    s_add_u32 s40, s59, s40
    v_add_u32_e32 v22, s40, v12
    s_mul_i32 s40, s2, 0x100
    v_and_b32_e32 v3, 31, v0
    v_add_u32_e32 v3, s40, v3
    s_mul_i32 s40, s5, 32
    v_add_u32_e32 v3, s40, v3
    v_lshlrev_b32_e32 v3, 2, v3
    v_mul_f32_e64 v12, v24, s28
    v_log_f32_e32 v13, v18
    v_cmp_eq_f32_e64 s[40:41], v18, 0
    s_nop 1
    v_rcp_f32_e32 v18, v18
    s_nop 1
    v_cndmask_b32_e64 v18, v18, 0, s[40:41]
    v_fma_f32 v2, v13, s45, v12
    s_mul_i32 s40, s79, 16
    v_add_u32_e32 v23, s40, v22
    v_mov_b32_e32 v19, v18
    v_pk_mul_f32 v[96:97], v[18:19], v[96:97]
    v_pk_mul_f32 v[98:99], v[18:19], v[98:99]
    v_pk_mul_f32 v[100:101], v[18:19], v[100:101]
    v_pk_mul_f32 v[102:103], v[18:19], v[102:103]
    v_pk_mul_f32 v[104:105], v[18:19], v[104:105]
    v_pk_mul_f32 v[106:107], v[18:19], v[106:107]
    v_pk_mul_f32 v[108:109], v[18:19], v[108:109]
    v_pk_mul_f32 v[110:111], v[18:19], v[110:111]
    v_cvt_pk_bf16_f32 v96, v96, v97
    v_cvt_pk_bf16_f32 v97, v98, v99
    v_cvt_pk_bf16_f32 v98, v100, v101
    v_cvt_pk_bf16_f32 v99, v102, v103
    v_nop
    v_permlane32_swap_b32_e32 v96, v98
    v_permlane32_swap_b32_e32 v97, v99
    v_cvt_pk_bf16_f32 v104, v104, v105
    v_cvt_pk_bf16_f32 v105, v106, v107
    v_cvt_pk_bf16_f32 v106, v108, v109
    v_cvt_pk_bf16_f32 v107, v110, v111
    v_nop
    v_permlane32_swap_b32_e32 v104, v106
    v_permlane32_swap_b32_e32 v105, v107
    v_permlane16_swap_b32_e32 v96, v104
    v_permlane16_swap_b32_e32 v97, v105
    v_permlane16_swap_b32_e32 v98, v106
    v_permlane16_swap_b32_e32 v99, v107
    buffer_store_dwordx4 v[96:99], v22, s[20:23], 0 offen
    buffer_store_dwordx4 v[104:107], v23, s[20:23], 0 offen
    v_pk_mul_f32 v[112:113], v[18:19], v[112:113]
    v_pk_mul_f32 v[114:115], v[18:19], v[114:115]
    v_pk_mul_f32 v[116:117], v[18:19], v[116:117]
    v_pk_mul_f32 v[118:119], v[18:19], v[118:119]
    v_pk_mul_f32 v[120:121], v[18:19], v[120:121]
    v_pk_mul_f32 v[122:123], v[18:19], v[122:123]
    v_pk_mul_f32 v[124:125], v[18:19], v[124:125]
    v_pk_mul_f32 v[126:127], v[18:19], v[126:127]
    v_cvt_pk_bf16_f32 v112, v112, v113
    v_cvt_pk_bf16_f32 v113, v114, v115
    v_cvt_pk_bf16_f32 v114, v116, v117
    v_cvt_pk_bf16_f32 v115, v118, v119
    v_nop
    v_permlane32_swap_b32_e32 v112, v114
    v_permlane32_swap_b32_e32 v113, v115
    v_cvt_pk_bf16_f32 v120, v120, v121
    v_cvt_pk_bf16_f32 v121, v122, v123
    v_cvt_pk_bf16_f32 v122, v124, v125
    v_cvt_pk_bf16_f32 v123, v126, v127
    v_nop
    v_permlane32_swap_b32_e32 v120, v122
    v_permlane32_swap_b32_e32 v121, v123
    v_permlane16_swap_b32_e32 v112, v120
    v_permlane16_swap_b32_e32 v113, v121
    v_permlane16_swap_b32_e32 v114, v122
    v_permlane16_swap_b32_e32 v115, v123
    buffer_store_dwordx4 v[112:115], v22, s[20:23], 0 offen offset:64
    buffer_store_dwordx4 v[120:123], v23, s[20:23], 0 offen offset:64
    v_pk_mul_f32 v[128:129], v[18:19], v[128:129]
    v_pk_mul_f32 v[130:131], v[18:19], v[130:131]
    v_pk_mul_f32 v[132:133], v[18:19], v[132:133]
    v_pk_mul_f32 v[134:135], v[18:19], v[134:135]
    v_pk_mul_f32 v[136:137], v[18:19], v[136:137]
    v_pk_mul_f32 v[138:139], v[18:19], v[138:139]
    v_pk_mul_f32 v[140:141], v[18:19], v[140:141]
    v_pk_mul_f32 v[142:143], v[18:19], v[142:143]
    v_cvt_pk_bf16_f32 v128, v128, v129
    v_cvt_pk_bf16_f32 v129, v130, v131
    v_cvt_pk_bf16_f32 v130, v132, v133
    v_cvt_pk_bf16_f32 v131, v134, v135
    v_nop
    v_permlane32_swap_b32_e32 v128, v130
    v_permlane32_swap_b32_e32 v129, v131
    v_cvt_pk_bf16_f32 v136, v136, v137
    v_cvt_pk_bf16_f32 v137, v138, v139
    v_cvt_pk_bf16_f32 v138, v140, v141
    v_cvt_pk_bf16_f32 v139, v142, v143
    v_nop
    v_permlane32_swap_b32_e32 v136, v138
    v_permlane32_swap_b32_e32 v137, v139
    v_permlane16_swap_b32_e32 v128, v136
    v_permlane16_swap_b32_e32 v129, v137
    v_permlane16_swap_b32_e32 v130, v138
    v_permlane16_swap_b32_e32 v131, v139
    buffer_store_dwordx4 v[128:131], v22, s[20:23], 0 offen offset:128
    buffer_store_dwordx4 v[136:139], v23, s[20:23], 0 offen offset:128
    v_pk_mul_f32 v[144:145], v[18:19], v[144:145]
    v_pk_mul_f32 v[146:147], v[18:19], v[146:147]
    v_pk_mul_f32 v[148:149], v[18:19], v[148:149]
    v_pk_mul_f32 v[150:151], v[18:19], v[150:151]
    v_pk_mul_f32 v[152:153], v[18:19], v[152:153]
    v_pk_mul_f32 v[154:155], v[18:19], v[154:155]
    v_pk_mul_f32 v[156:157], v[18:19], v[156:157]
    v_pk_mul_f32 v[158:159], v[18:19], v[158:159]
    v_cvt_pk_bf16_f32 v144, v144, v145
    v_cvt_pk_bf16_f32 v145, v146, v147
    v_cvt_pk_bf16_f32 v146, v148, v149
    v_cvt_pk_bf16_f32 v147, v150, v151
    v_nop
    v_permlane32_swap_b32_e32 v144, v146
    v_permlane32_swap_b32_e32 v145, v147
    v_cvt_pk_bf16_f32 v152, v152, v153
    v_cvt_pk_bf16_f32 v153, v154, v155
    v_cvt_pk_bf16_f32 v154, v156, v157
    v_cvt_pk_bf16_f32 v155, v158, v159
    v_nop
    v_permlane32_swap_b32_e32 v152, v154
    v_permlane32_swap_b32_e32 v153, v155
    v_permlane16_swap_b32_e32 v144, v152
    v_permlane16_swap_b32_e32 v145, v153
    v_permlane16_swap_b32_e32 v146, v154
    v_permlane16_swap_b32_e32 v147, v155
    buffer_store_dwordx4 v[144:147], v22, s[20:23], 0 offen offset:192
    buffer_store_dwordx4 v[152:155], v23, s[20:23], 0 offen offset:192
    s_cmp_eq_u32 s56, 0
    s_cbranch_scc1 label_0FF8
    v_cmp_ge_f32_e64 s[40:41], v2, v27
    v_cndmask_b32_e64 v2, v27, v2, s[40:41]
    buffer_store_dword v2, v3, s[24:27], 0 offen
    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    s_endpgm

.size _ZN5aiter19fmha_fwd_hd128_bf16E, .-_ZN5aiter19fmha_fwd_hd128_bf16E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter19fmha_fwd_hd128_bf16E
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 512
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 160
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
  - .name: _ZN5aiter19fmha_fwd_hd128_bf16E
    .symbol: _ZN5aiter19fmha_fwd_hd128_bf16E.kd
    .kernarg_segment_size: 512
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 256
    .agpr_count: 96
    .max_flat_workgroup_size: 256
    .args:
      - .name: R
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: Q
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
      - .name: K
        .size: 8
        .offset: 32
        .value_kind: global_buffer
        .address_space: global
      - .name: V
        .size: 8
        .offset: 48
        .value_kind: global_buffer
        .address_space: global
      - .name: LSE
        .size: 8
        .offset: 64
        .value_kind: global_buffer
        .address_space: global
      - .name: scalar
        .size: 4
        .offset: 80
        .value_kind: by_value
      - .name: seq_len
        .size: 4
        .offset: 96
        .value_kind: by_value
      - .name: head_dim
        .size: 4
        .offset: 112
        .value_kind: by_value
...
.end_amdgpu_metadata

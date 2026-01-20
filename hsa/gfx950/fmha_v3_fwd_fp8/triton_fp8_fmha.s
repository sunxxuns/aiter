	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	_attn_fwd                       ; -- Begin function _attn_fwd
	.p2align	8
	.type	_attn_fwd,@function
_attn_fwd:                              ; @_attn_fwd
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.9:
	.file	1 "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8" "bench_triton_fmha.py"
	.loc	1 46 0 prologue_end             ; bench_triton_fmha.py:46:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.10:
.LBB0_0:
	s_load_dwordx2 s[18:19], s[0:1], 0x60
	s_load_dword s24, s[0:1], 0x38
.Ltmp1:
	.loc	1 59 22 is_stmt 1               ; bench_triton_fmha.py:59:22
	s_abs_i32 s21, s17
	.loc	1 95 46                         ; bench_triton_fmha.py:95:46
	v_lshrrev_b32_e32 v3, 3, v0
	v_lshlrev_b32_e32 v28, 4, v0
	.loc	1 59 22                         ; bench_triton_fmha.py:59:22
	s_waitcnt lgkmcnt(0)
	s_abs_i32 s9, s18
	v_cvt_f32_u32_e32 v1, s9
	s_sub_i32 s22, 0, s9
	s_xor_b32 s20, s17, s18
	s_ashr_i32 s20, s20, 31
	v_rcp_iflag_f32_e32 v1, v1
	.loc	1 95 46                         ; bench_triton_fmha.py:95:46
	v_and_b32_e32 v2, 0x70, v28
	v_lshlrev_b32_e32 v141, 2, v0
	.loc	1 59 22                         ; bench_triton_fmha.py:59:22
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s23, v1
	s_mul_i32 s22, s22, s23
	s_mul_hi_u32 s22, s23, s22
	s_add_i32 s23, s23, s22
	s_mul_hi_u32 s22, s21, s23
	s_mul_i32 s23, s22, s9
	s_sub_i32 s21, s21, s23
	s_add_i32 s23, s22, 1
	s_sub_i32 s25, s21, s9
	s_cmp_ge_u32 s21, s9
	s_cselect_b32 s22, s23, s22
	s_cselect_b32 s21, s25, s21
	s_add_i32 s23, s22, 1
	s_cmp_ge_u32 s21, s9
	s_cselect_b32 s9, s23, s22
	s_xor_b32 s9, s9, s20
	s_sub_i32 s9, s9, s20
	.loc	1 60 21                         ; bench_triton_fmha.py:60:21
	s_mul_i32 s18, s9, s18
	s_sub_i32 s18, s17, s18
	.loc	1 61 38                         ; bench_triton_fmha.py:61:38
	s_mul_hi_i32 s21, s9, s14
	s_mul_i32 s20, s9, s14
	.loc	1 61 71 is_stmt 0               ; bench_triton_fmha.py:61:71
	s_mul_hi_i32 s9, s18, s15
	s_mul_i32 s18, s18, s15
	.loc	1 61 50                         ; bench_triton_fmha.py:61:50
	s_add_u32 s14, s18, s20
	s_addc_u32 s15, s9, s21
	.loc	1 64 17 is_stmt 1               ; bench_triton_fmha.py:64:17
	s_add_u32 s25, s2, s14
	s_addc_u32 s26, s3, s15
	.loc	1 67 27                         ; bench_triton_fmha.py:67:27
	s_lshl_b32 s2, s16, 7
	.loc	1 95 46                         ; bench_triton_fmha.py:95:46
	v_or_b32_e32 v1, 0x60, v3
	.loc	1 69 8                          ; bench_triton_fmha.py:69:8
	s_ashr_i32 s3, s2, 31
	.loc	1 101 16                        ; bench_triton_fmha.py:101:16
	s_mul_i32 s27, s24, s2
	s_lshl_b32 s28, s24, 5
	v_mad_u64_u32 v[4:5], s[22:23], s24, v3, v[2:3]
	v_mad_u64_u32 v[10:11], s[22:23], s24, v1, v[2:3]
	s_mul_hi_i32 s16, s24, s2
	v_add_u32_e32 v6, s28, v4
	s_add_u32 s22, s25, s27
	v_add_u32_e32 v8, s28, v6
	s_addc_u32 s23, s26, s16
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[20:21], s[22:23], 0, v[4:5]
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[22:23], s[22:23], 0, v[6:7]
	v_lshl_add_u64 v[24:25], s[22:23], 0, v[8:9]
	v_lshl_add_u64 v[26:27], s[22:23], 0, v[10:11]
	global_load_dwordx4 v[4:7], v[20:21], off
	global_load_dwordx4 v[8:11], v[22:23], off
	global_load_dwordx4 v[12:15], v[24:25], off
	global_load_dwordx4 v[16:19], v[26:27], off
	s_movk_i32 s16, 0x70
	v_bitop3_b32 v1, v28, v0, s16 bitop3:0x78
	v_add_u32_e32 v142, 0, v1
.Ltmp2:
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	s_cmp_gt_i32 s19, 0
.Ltmp3:
	.loc	1 101 16                        ; bench_triton_fmha.py:101:16
	s_waitcnt vmcnt(3)
	ds_write_b128 v142, v[4:7]
	s_waitcnt vmcnt(2)
	ds_write_b128 v142, v[8:11] offset:4096
	s_waitcnt vmcnt(1)
	ds_write_b128 v142, v[12:15] offset:8192
	s_waitcnt vmcnt(0)
	ds_write_b128 v142, v[16:19] offset:12288
	s_waitcnt lgkmcnt(0)
	s_barrier
.Ltmp4:
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	s_cbranch_scc1 .LBB0_2
.Ltmp5:
; %bb.1:                                ; %.._crit_edge_crit_edge
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	v_lshlrev_b32_e32 v67, 2, v0
	s_mov_b64 s[22:23], 0
	s_branch .LBB0_3
.LBB0_2:
	.loc	1 0 21 is_stmt 0                ; bench_triton_fmha.py:0:21
	s_mov_b64 s[22:23], -1
                                        ; implicit-def: $vgpr67
.LBB0_3:                                ; %Flow
	v_and_b32_e32 v1, 31, v0
	v_and_b32_e32 v138, 0xc0, v0
	s_andn2_b64 vcc, exec, s[22:23]
	v_and_b32_e32 v140, 32, v0
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	.loc	1 101 16 is_stmt 1              ; bench_triton_fmha.py:101:16
	v_lshlrev_b32_e32 v10, 3, v0
	.loc	1 100 26                        ; bench_triton_fmha.py:100:26
	v_mov_b32_e32 v4, 0x3fb8aa3b
	.loc	1 101 16                        ; bench_triton_fmha.py:101:16
	v_lshlrev_b32_e32 v8, 7, v1
	v_and_b32_e32 v9, 0x70, v10
	s_load_dword s22, s[0:1], 0x50
	s_load_dword s24, s[0:1], 0x44
	.loc	1 100 26                        ; bench_triton_fmha.py:100:26
	v_mul_f32_e32 v143, s8, v4
	.loc	1 101 16                        ; bench_triton_fmha.py:101:16
	v_lshlrev_b32_e32 v4, 6, v138
	v_lshrrev_b32_e32 v11, 1, v140
	v_or_b32_e32 v5, v8, v9
	v_bitop3_b32 v4, v5, v11, v4 bitop3:0x36
	v_xor_b32_e32 v5, 0x60, v4
	v_add_u32_e32 v5, 0, v5
	v_xad_u32 v6, v4, 64, 0
	ds_read_b128 v[118:121], v5
	ds_read_b128 v[114:117], v6
	v_xad_u32 v5, v4, 32, 0
	v_add_u32_e32 v4, 0, v4
	.loc	1 77 8                          ; bench_triton_fmha.py:77:8
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s23, s22, 31
	.loc	1 85 8                          ; bench_triton_fmha.py:85:8
	s_ashr_i32 s25, s24, 31
	.loc	1 101 16                        ; bench_triton_fmha.py:101:16
	ds_read_b128 v[126:129], v5
	ds_read_b128 v[122:125], v4
	v_mad_u64_u32 v[4:5], s[26:27], s24, v3, v[2:3]
	v_mad_u64_u32 v[2:3], s[26:27], s22, v3, v[2:3]
	v_lshlrev_b32_e32 v14, 6, v0
	v_and_b32_e32 v15, 48, v141
	s_movk_i32 s8, 0xb80
.Ltmp6:
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	s_add_u32 s26, s6, s18
	v_and_or_b32 v14, v14, s8, v15
	v_and_b32_e32 v10, 8, v10
	v_and_b32_e32 v15, 16, v0
	s_addc_u32 s27, s7, s9
	s_lshl_b64 s[6:7], s[22:23], 6
	v_lshl_add_u32 v6, s24, 5, v4
	v_bitop3_b32 v144, v8, v11, v9 bitop3:0x36
	v_lshl_add_u32 v8, s22, 5, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_bitop3_b32 v10, v14, v15, v10 bitop3:0x36
	s_add_u32 s8, s4, s18
.Ltmp7:
	.loc	1 101 16                        ; bench_triton_fmha.py:101:16
	s_mov_b32 s36, 0
	v_ashrrev_i32_e32 v5, 31, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_xor_b32_e32 v11, 32, v144
	v_xor_b32_e32 v12, 64, v144
	v_xor_b32_e32 v13, 0x60, v144
	v_ashrrev_i32_e32 v9, 31, v8
	v_xor_b32_e32 v14, 32, v10
	v_xor_b32_e32 v15, 0x460, v10
	v_xor_b32_e32 v16, 0x1020, v10
	v_xor_b32_e32 v17, 0x1460, v10
	v_xor_b32_e32 v18, 0x60, v10
	v_xor_b32_e32 v19, 0x420, v10
	v_xor_b32_e32 v20, 0x1060, v10
	v_xor_b32_e32 v21, 0x1420, v10
.Ltmp8:
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	v_lshl_add_u64 v[132:133], s[26:27], 0, v[2:3]
	s_addc_u32 s9, s5, s9
	v_mov_b32_e32 v2, 0
	v_cmp_eq_u32_e32 vcc, 0, v140
	v_add_u32_e32 v145, 0, v10
	v_lshl_add_u64 v[130:131], s[26:27], 0, v[8:9]
	v_lshl_add_u64 v[134:135], s[8:9], 0, v[6:7]
	s_lshl_b64 s[4:5], s[24:25], 6
	v_lshl_add_u64 v[136:137], s[8:9], 0, v[4:5]
	v_add_u32_e32 v146, 0, v11
	v_add_u32_e32 v147, 0, v12
	v_add_u32_e32 v148, 0, v13
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	v_add_u32_e32 v149, 0, v14
	v_add_u32_e32 v150, 0, v15
	v_add_u32_e32 v151, 0, v16
	v_add_u32_e32 v152, 0, v17
	v_add_u32_e32 v153, 0, v18
	v_add_u32_e32 v154, 0, v19
	v_add_u32_e32 v155, 0, v20
	v_add_u32_e32 v156, 0, v21
	s_mov_b32 s8, s36
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_xor_b32_e32 v158, 0x80, v141
	v_mov_b32_e32 v139, 1.0
	v_mov_b32_e32 v157, 0xff800000
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 27 20                         ; bench_triton_fmha.py:27:20 @[ bench_triton_fmha.py:105:60 ]
	v_lshl_add_u64 v[66:67], v[136:137], 0, s[20:21]
	v_lshl_add_u64 v[70:71], v[134:135], 0, s[20:21]
	global_load_dwordx4 v[66:69], v[66:67], off
	s_nop 0
	global_load_dwordx4 v[70:73], v[70:71], off
	v_add_u32_e32 v74, 0, v144
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 28 23                         ; bench_triton_fmha.py:28:23 @[ bench_triton_fmha.py:105:60 ]
	v_mov_b64_e32 v[112:113], s[50:51]
	v_mov_b64_e32 v[110:111], s[48:49]
	v_mov_b64_e32 v[108:109], s[46:47]
	v_mov_b64_e32 v[106:107], s[44:45]
	v_mov_b64_e32 v[104:105], s[42:43]
	v_mov_b64_e32 v[102:103], s[40:41]
	v_mov_b64_e32 v[100:101], s[38:39]
	v_mov_b64_e32 v[98:99], s[36:37]
	v_mov_b32_e32 v159, v139
	v_mov_b32_e32 v139, v157
	.loc	1 29 31                         ; bench_triton_fmha.py:29:31 @[ bench_triton_fmha.py:105:60 ]
	v_max_f32_e32 v157, v139, v139
	.loc	1 27 20                         ; bench_triton_fmha.py:27:20 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt vmcnt(1)
	ds_write_b128 v142, v[66:69]
	s_waitcnt vmcnt(0)
	ds_write_b128 v142, v[70:73] offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[70:73], v146
	ds_read_b128 v[66:69], v74
	ds_read_b128 v[160:163], v74 offset:4096
	ds_read_b128 v[164:167], v146 offset:4096
	.loc	1 28 23                         ; bench_triton_fmha.py:28:23 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x64_f8f6f4 v[82:97], v[66:73], v[122:129], v[98:113]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[66:81], v[160:167], v[122:129], v[98:113]
	.loc	1 27 20                         ; bench_triton_fmha.py:27:20 @[ bench_triton_fmha.py:105:60 ]
	s_nop 7
	s_nop 6
	ds_read_b128 v[102:105], v148
	ds_read_b128 v[98:101], v147
	ds_read_b128 v[106:109], v147 offset:4096
	ds_read_b128 v[110:113], v148 offset:4096
	.loc	1 28 23                         ; bench_triton_fmha.py:28:23 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x64_f8f6f4 v[82:97], v[98:105], v[114:121], v[82:97]
	.loc	1 36 20                         ; bench_triton_fmha.py:36:20 @[ bench_triton_fmha.py:105:60 ]
	v_lshl_add_u64 v[98:99], v[132:133], 0, s[20:21]
	v_lshl_add_u64 v[102:103], v[130:131], 0, s[20:21]
	global_load_dwordx4 v[98:101], v[98:99], off
	s_nop 0
	global_load_dwordx4 v[102:105], v[102:103], off
	.loc	1 28 23                         ; bench_triton_fmha.py:28:23 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[66:81], v[106:113], v[114:121], v[66:81]
	.file	2 "/opt/venv/lib/python3.10/site-packages/triton/language" "standard.py"
	.loc	2 168 27                        ; standard.py:168:27 @[ bench_triton_fmha.py:105:60 ]
	s_nop 7
	s_nop 4
	v_max_f32_e32 v106, v83, v83
	v_max_f32_e32 v107, v82, v82
	v_max_f32_e32 v106, v107, v106
	v_max3_f32 v106, v106, v84, v85
	v_max3_f32 v106, v106, v86, v87
	v_max3_f32 v106, v106, v88, v89
	v_max3_f32 v106, v106, v90, v91
	v_max3_f32 v106, v106, v92, v93
	v_max3_f32 v106, v106, v94, v95
	v_max3_f32 v106, v106, v96, v97
	v_max3_f32 v106, v106, v66, v67
	v_max3_f32 v106, v106, v68, v69
	v_max3_f32 v106, v106, v70, v71
	v_max3_f32 v106, v106, v72, v73
	v_max3_f32 v106, v106, v74, v75
	v_max3_f32 v106, v106, v76, v77
	v_max3_f32 v106, v106, v78, v79
	v_max3_f32 v106, v106, v80, v81
	.loc	2 189 40                        ; standard.py:189:40 @[ bench_triton_fmha.py:105:60 ]
	v_mov_b32_e32 v107, v106
	s_nop 1
	v_permlane32_swap_b32_e32 v106, v107
	.loc	2 168 27                        ; standard.py:168:27 @[ bench_triton_fmha.py:105:60 ]
	v_max_f32_e32 v107, v107, v107
	v_max_f32_e32 v106, v106, v106
	v_max_f32_e32 v106, v106, v107
	.loc	1 29 47                         ; bench_triton_fmha.py:29:47 @[ bench_triton_fmha.py:105:60 ]
	v_mul_f32_e32 v106, v143, v106
	.loc	1 29 31 is_stmt 0               ; bench_triton_fmha.py:29:31 @[ bench_triton_fmha.py:105:60 ]
	v_max_f32_e32 v157, v157, v106
	.loc	1 30 29 is_stmt 1               ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v82, v143, v82, -v157
	v_fma_f32 v83, v143, v83, -v157
	v_fma_f32 v84, v143, v84, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v82, v82
	v_exp_f32_e32 v83, v83
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v85, v143, v85, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v84, v84
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v86, v143, v86, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v85, v85
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v87, v143, v87, -v157
	v_fma_f32 v66, v143, v66, -v157
	.loc	1 33 35                         ; bench_triton_fmha.py:33:35 @[ bench_triton_fmha.py:105:60 ]
	v_sub_f32_e32 v106, v139, v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v86, v86
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v88, v143, v88, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v107, v66
	.loc	1 33 29                         ; bench_triton_fmha.py:33:29 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v66, v106
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v82, v83
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v89, v143, v89, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v88, v88
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v84, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v90, v143, v90, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v89, v89
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v85, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v91, v143, v91, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v90, v90
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v86, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v92, v143, v92, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v91, v91
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v87, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v93, v143, v93, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v92, v92
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v88, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v94, v143, v94, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v93, v93
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v89, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v95, v143, v95, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v94, v94
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v90, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v96, v143, v96, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v95, v95
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v91, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v97, v143, v97, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v96, v96
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v92, v106
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v97, v97
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v93, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v67, v143, v67, -v157
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v94, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v68, v143, v68, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v67, v67
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v95, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v69, v143, v69, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v68, v68
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v96, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v70, v143, v70, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v69, v69
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v97, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v71, v143, v71, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v70, v70
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v107, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v72, v143, v72, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v71, v71
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v67, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v73, v143, v73, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v72, v72
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v68, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v74, v143, v74, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v73, v73
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v69, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v75, v143, v75, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v74, v74
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v70, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v76, v143, v76, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v75, v75
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v71, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v77, v143, v77, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v76, v76
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v72, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v78, v143, v78, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v77, v77
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v73, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v79, v143, v79, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v78, v78
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v74, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v80, v143, v80, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v79, v79
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v75, v106
	.loc	1 30 29                         ; bench_triton_fmha.py:30:29 @[ bench_triton_fmha.py:105:60 ]
	v_fma_f32 v81, v143, v81, -v157
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v80, v80
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v76, v106
	.loc	1 31 25                         ; bench_triton_fmha.py:31:25 @[ bench_triton_fmha.py:105:60 ]
	v_exp_f32_e32 v81, v81
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v106, v77, v106
	v_add_f32_e32 v106, v78, v106
	v_add_f32_e32 v106, v79, v106
	v_add_f32_e32 v106, v80, v106
	v_add_f32_e32 v106, v81, v106
	.loc	2 291 36                        ; standard.py:291:36 @[ bench_triton_fmha.py:105:60 ]
	v_mov_b32_e32 v108, v106
	s_nop 1
	v_permlane32_swap_b32_e32 v106, v108
	.loc	2 261 15                        ; standard.py:261:15 @[ bench_triton_fmha.py:105:60 ]
	v_add_f32_e32 v139, v106, v108
	.loc	1 35 20                         ; bench_triton_fmha.py:35:20 @[ bench_triton_fmha.py:105:60 ]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[66:67] op_sel_hi:[1,0]
	.loc	1 34 28                         ; bench_triton_fmha.py:34:28 @[ bench_triton_fmha.py:105:60 ]
	v_fmac_f32_e32 v139, v159, v66
	.loc	1 37 17                         ; bench_triton_fmha.py:37:17 @[ bench_triton_fmha.py:105:60 ]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v66, v82, v83, 1.0
	v_cvt_scalef32_pk_fp8_f32 v66, v84, v85, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v82, v86, v87, 1.0
	v_cvt_scalef32_pk_fp8_f32 v82, v88, v89, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v83, v90, v91, 1.0
	v_cvt_scalef32_pk_fp8_f32 v83, v92, v93, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v84, v94, v95, 1.0
	v_cvt_scalef32_pk_fp8_f32 v84, v96, v97, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v85, v107, v67, 1.0
	v_cvt_scalef32_pk_fp8_f32 v85, v68, v69, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v86, v70, v71, 1.0
	v_cvt_scalef32_pk_fp8_f32 v86, v72, v73, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cvt_scalef32_pk_fp8_f32 v71, v74, v75, 1.0
	v_cvt_scalef32_pk_fp8_f32 v71, v76, v77, 1.0 op_sel:[0,0,0,1]
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1
	v_cndmask_b32_e32 v67, v66, v83, vcc
	ds_bpermute_b32 v67, v158, v67
	v_cndmask_b32_e32 v68, v82, v84, vcc
	v_cvt_scalef32_pk_fp8_f32 v73, v78, v79, 1.0
	ds_bpermute_b32 v69, v158, v68
	v_cvt_scalef32_pk_fp8_f32 v73, v80, v81, 1.0 op_sel:[0,0,0,1]
	v_cndmask_b32_e32 v68, v85, v71, vcc
	ds_bpermute_b32 v72, v158, v68
	v_cndmask_b32_e32 v68, v86, v73, vcc
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e32 v66, v67, v66, vcc
	v_cndmask_b32_e32 v67, v83, v67, vcc
	ds_bpermute_b32 v83, v158, v68
	.loc	1 36 20                         ; bench_triton_fmha.py:36:20 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b128 v142, v[98:101]
	s_waitcnt vmcnt(0)
	ds_write_b128 v142, v[102:105] offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b8 v[74:75], v145
	ds_read_b64_tr_b8 v[76:77], v145 offset:1088
	ds_read_b64_tr_b8 v[78:79], v145 offset:4096
	ds_read_b64_tr_b8 v[80:81], v145 offset:5184
	.loc	1 37 17                         ; bench_triton_fmha.py:37:17 @[ bench_triton_fmha.py:105:60 ]
	v_cndmask_b32_e32 v68, v69, v82, vcc
	v_cndmask_b32_e32 v69, v84, v69, vcc
	v_cndmask_b32_e32 v70, v72, v85, vcc
	v_cndmask_b32_e32 v71, v71, v72, vcc
	v_cndmask_b32_e32 v72, v83, v86, vcc
	v_cndmask_b32_e32 v73, v73, v83, vcc
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	s_add_i32 s8, s8, 64
	v_lshl_add_u64 v[130:131], v[130:131], 0, s[6:7]
	.loc	1 38 27                         ; bench_triton_fmha.py:38:27 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[50:65], v[74:81], v[66:73], v[50:65]
	.loc	1 36 20                         ; bench_triton_fmha.py:36:20 @[ bench_triton_fmha.py:105:60 ]
	ds_read_b64_tr_b8 v[74:75], v149
	ds_read_b64_tr_b8 v[76:77], v150
	ds_read_b64_tr_b8 v[78:79], v151
	ds_read_b64_tr_b8 v[80:81], v152
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	v_lshl_add_u64 v[132:133], v[132:133], 0, s[6:7]
	v_lshl_add_u64 v[134:135], v[134:135], 0, s[4:5]
	v_lshl_add_u64 v[136:137], v[136:137], 0, s[4:5]
	s_cmp_gt_i32 s19, s8
	.loc	1 38 27                         ; bench_triton_fmha.py:38:27 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[34:49], v[74:81], v[66:73], v[34:49]
	.loc	1 36 20                         ; bench_triton_fmha.py:36:20 @[ bench_triton_fmha.py:105:60 ]
	ds_read_b64_tr_b8 v[74:75], v145 offset:64
	ds_read_b64_tr_b8 v[76:77], v145 offset:1024
	ds_read_b64_tr_b8 v[78:79], v145 offset:4160
	ds_read_b64_tr_b8 v[80:81], v145 offset:5120
	.loc	1 38 27                         ; bench_triton_fmha.py:38:27 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[18:33], v[74:81], v[66:73], v[18:33]
	.loc	1 36 20                         ; bench_triton_fmha.py:36:20 @[ bench_triton_fmha.py:105:60 ]
	ds_read_b64_tr_b8 v[74:75], v153
	ds_read_b64_tr_b8 v[76:77], v154
	ds_read_b64_tr_b8 v[78:79], v155
	ds_read_b64_tr_b8 v[80:81], v156
	.loc	1 38 27                         ; bench_triton_fmha.py:38:27 @[ bench_triton_fmha.py:105:60 ]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[2:17], v[74:81], v[66:73], v[2:17]
	.loc	1 25 33                         ; bench_triton_fmha.py:25:33 @[ bench_triton_fmha.py:105:60 ]
	s_cbranch_scc1 .LBB0_5
.Ltmp9:
; %bb.6:                                ; %._crit_edge.loopexit
	.loc	1 0 33 is_stmt 0                ; bench_triton_fmha.py:0:33
	v_mov_b32_e32 v67, v141
	.loc	1 95 46 is_stmt 1               ; bench_triton_fmha.py:95:46
	s_branch .LBB0_8
.LBB0_7:
	.loc	1 0 46 is_stmt 0                ; bench_triton_fmha.py:0:46
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v139, 1.0
	v_mov_b32_e32 v157, 0xff800000
.LBB0_8:                                ; %Flow175
	.loc	1 106 24 is_stmt 1              ; bench_triton_fmha.py:106:24
	v_log_f32_e32 v69, v139
	.loc	1 107 16                        ; bench_triton_fmha.py:107:16
	v_div_scale_f32 v70, s[6:7], v139, v139, v50
	v_rcp_f32_e32 v71, v70
	.loc	1 106 11                        ; bench_triton_fmha.py:106:11
	v_add_f32_e32 v69, v157, v69
	.loc	1 107 16                        ; bench_triton_fmha.py:107:16
	v_div_scale_f32 v72, vcc, v50, v139, v50
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v50, v70, v139, v50
	v_div_scale_f32 v70, s[6:7], v139, v139, v51
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v51, v139, v51
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v51, v70, v139, v51
	v_div_scale_f32 v70, s[6:7], v139, v139, v52
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v52, v139, v52
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v52, v70, v139, v52
	v_div_scale_f32 v70, s[6:7], v139, v139, v53
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v53, v139, v53
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v53, v70, v139, v53
	v_div_scale_f32 v70, s[6:7], v139, v139, v54
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v54, v139, v54
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v54, v70, v139, v54
	v_div_scale_f32 v70, s[6:7], v139, v139, v55
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v55, v139, v55
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v55, v70, v139, v55
	v_div_scale_f32 v70, s[6:7], v139, v139, v56
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v56, v139, v56
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v56, v70, v139, v56
	v_div_scale_f32 v70, s[6:7], v139, v139, v57
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v57, v139, v57
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v57, v70, v139, v57
	v_div_scale_f32 v70, s[6:7], v139, v139, v58
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v58, v139, v58
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v58, v70, v139, v58
	v_div_scale_f32 v70, s[6:7], v139, v139, v59
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v59, v139, v59
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v59, v70, v139, v59
	v_div_scale_f32 v70, s[6:7], v139, v139, v60
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v60, v139, v60
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v60, v70, v139, v60
	v_div_scale_f32 v70, s[6:7], v139, v139, v61
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v61, v139, v61
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v61, v70, v139, v61
	v_div_scale_f32 v70, s[6:7], v139, v139, v62
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v62, v139, v62
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v62, v70, v139, v62
	v_div_scale_f32 v70, s[6:7], v139, v139, v63
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v63, v139, v63
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v63, v70, v139, v63
	v_div_scale_f32 v70, s[6:7], v139, v139, v64
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v64, v139, v64
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v64, v70, v139, v64
	v_div_scale_f32 v70, s[6:7], v139, v139, v65
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v65, v139, v65
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v65, v70, v139, v65
	v_div_scale_f32 v70, s[6:7], v139, v139, v34
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v34, v139, v34
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v34, v70, v139, v34
	v_div_scale_f32 v70, s[6:7], v139, v139, v35
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v35, v139, v35
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v35, v70, v139, v35
	v_div_scale_f32 v70, s[6:7], v139, v139, v36
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v36, v139, v36
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v36, v70, v139, v36
	v_div_scale_f32 v70, s[6:7], v139, v139, v37
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v37, v139, v37
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v37, v70, v139, v37
	v_div_scale_f32 v70, s[6:7], v139, v139, v38
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v38, v139, v38
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v38, v70, v139, v38
	v_div_scale_f32 v70, s[6:7], v139, v139, v39
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v39, v139, v39
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v39, v70, v139, v39
	v_div_scale_f32 v70, s[6:7], v139, v139, v40
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v40, v139, v40
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v40, v70, v139, v40
	v_div_scale_f32 v70, s[6:7], v139, v139, v41
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v41, v139, v41
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v41, v70, v139, v41
	v_div_scale_f32 v70, s[6:7], v139, v139, v42
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v42, v139, v42
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v42, v70, v139, v42
	v_div_scale_f32 v70, s[6:7], v139, v139, v43
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v43, v139, v43
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v43, v70, v139, v43
	v_div_scale_f32 v70, s[6:7], v139, v139, v44
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v44, v139, v44
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v44, v70, v139, v44
	v_div_scale_f32 v70, s[6:7], v139, v139, v45
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v45, v139, v45
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v45, v70, v139, v45
	v_div_scale_f32 v70, s[6:7], v139, v139, v46
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v46, v139, v46
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v46, v70, v139, v46
	v_div_scale_f32 v70, s[6:7], v139, v139, v47
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v47, v139, v47
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v47, v70, v139, v47
	v_div_scale_f32 v70, s[6:7], v139, v139, v48
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v48, v139, v48
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v48, v70, v139, v48
	v_div_scale_f32 v70, s[6:7], v139, v139, v49
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v49, v139, v49
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v49, v70, v139, v49
	v_div_scale_f32 v70, s[6:7], v139, v139, v18
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v18, v139, v18
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v18, v70, v139, v18
	v_div_scale_f32 v70, s[6:7], v139, v139, v19
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v19, v139, v19
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v19, v70, v139, v19
	v_div_scale_f32 v70, s[6:7], v139, v139, v20
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v20, v139, v20
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v20, v70, v139, v20
	v_div_scale_f32 v70, s[6:7], v139, v139, v21
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v21, v139, v21
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v21, v70, v139, v21
	v_div_scale_f32 v70, s[6:7], v139, v139, v22
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v22, v139, v22
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v22, v70, v139, v22
	v_div_scale_f32 v70, s[6:7], v139, v139, v23
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v23, v139, v23
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v23, v70, v139, v23
	v_div_scale_f32 v70, s[6:7], v139, v139, v24
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v24, v139, v24
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v24, v70, v139, v24
	v_div_scale_f32 v70, s[6:7], v139, v139, v25
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v25, v139, v25
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v25, v70, v139, v25
	v_div_scale_f32 v70, s[6:7], v139, v139, v26
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v26, v139, v26
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v26, v70, v139, v26
	v_div_scale_f32 v70, s[6:7], v139, v139, v27
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v27, v139, v27
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v27, v70, v139, v27
	v_div_scale_f32 v70, s[6:7], v139, v139, v28
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v28, v139, v28
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v28, v70, v139, v28
	v_div_scale_f32 v70, s[6:7], v139, v139, v29
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v29, v139, v29
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v29, v70, v139, v29
	v_div_scale_f32 v70, s[6:7], v139, v139, v30
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v30, v139, v30
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v30, v70, v139, v30
	v_div_scale_f32 v70, s[6:7], v139, v139, v31
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v31, v139, v31
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v31, v70, v139, v31
	v_div_scale_f32 v70, s[6:7], v139, v139, v32
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v32, v139, v32
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v32, v70, v139, v32
	v_div_scale_f32 v70, s[6:7], v139, v139, v33
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v33, v139, v33
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v33, v70, v139, v33
	v_div_scale_f32 v70, s[6:7], v139, v139, v2
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v2, v139, v2
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v2, v70, v139, v2
	v_div_scale_f32 v70, s[6:7], v139, v139, v3
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v3, v139, v3
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v3, v70, v139, v3
	v_div_scale_f32 v70, s[6:7], v139, v139, v4
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v4, v139, v4
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v4, v70, v139, v4
	v_div_scale_f32 v70, s[6:7], v139, v139, v5
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v5, v139, v5
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v5, v70, v139, v5
	v_div_scale_f32 v70, s[6:7], v139, v139, v6
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v6, v139, v6
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v6, v70, v139, v6
	v_div_scale_f32 v70, s[6:7], v139, v139, v7
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v7, v139, v7
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v7, v70, v139, v7
	v_div_scale_f32 v70, s[6:7], v139, v139, v8
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v8, v139, v8
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v8, v70, v139, v8
	v_div_scale_f32 v70, s[6:7], v139, v139, v9
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v9, v139, v9
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v9, v70, v139, v9
	v_div_scale_f32 v70, s[6:7], v139, v139, v10
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v10, v139, v10
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v10, v70, v139, v10
	v_div_scale_f32 v70, s[6:7], v139, v139, v11
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v11, v139, v11
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v11, v70, v139, v11
	v_div_scale_f32 v70, s[6:7], v139, v139, v12
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v12, v139, v12
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v12, v70, v139, v12
	v_div_scale_f32 v70, s[6:7], v139, v139, v13
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v13, v139, v13
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v13, v70, v139, v13
	v_div_scale_f32 v70, s[6:7], v139, v139, v14
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v14, v139, v14
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v14, v70, v139, v14
	v_div_scale_f32 v70, s[6:7], v139, v139, v15
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v15, v139, v15
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v15, v70, v139, v15
	v_div_scale_f32 v70, s[6:7], v139, v139, v16
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v16, v139, v16
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	v_div_fmas_f32 v70, v70, v71, v73
	v_div_fixup_f32 v16, v70, v139, v16
	v_div_scale_f32 v70, s[6:7], v139, v139, v17
	v_rcp_f32_e32 v71, v70
	v_div_scale_f32 v72, vcc, v17, v139, v17
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3
	v_fma_f32 v73, -v70, v71, 1.0
	v_fmac_f32_e32 v71, v73, v71
	v_mul_f32_e32 v73, v72, v71
	v_fma_f32 v74, -v70, v73, v72
	v_fmac_f32_e32 v73, v74, v71
	v_fma_f32 v70, -v70, v73, v72
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
	.loc	1 95 46                         ; bench_triton_fmha.py:95:46
	v_lshrrev_b32_e32 v68, 1, v138
	.loc	1 88 19                         ; bench_triton_fmha.py:88:19
	s_lshl_b64 s[4:5], s[14:15], 2
	.loc	1 107 16                        ; bench_triton_fmha.py:107:16
	v_div_fmas_f32 v70, v70, v71, v73
	s_load_dword s0, s[0:1], 0x5c
	.loc	1 95 46                         ; bench_triton_fmha.py:95:46
	v_or_b32_e32 v68, v68, v1
	.loc	1 88 19                         ; bench_triton_fmha.py:88:19
	s_add_u32 s1, s12, s4
	.loc	1 107 16                        ; bench_triton_fmha.py:107:16
	v_div_fixup_f32 v17, v70, v139, v17
	.loc	1 108 26                        ; bench_triton_fmha.py:108:26
	s_mul_i32 s6, s19, s17
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshlrev_b32_e32 v70, 1, v138
	.loc	1 88 19                         ; bench_triton_fmha.py:88:19
	s_addc_u32 s4, s13, s5
	.loc	1 108 17                        ; bench_triton_fmha.py:108:17
	s_ashr_i32 s7, s6, 31
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	v_add3_u32 v1, 0, v1, v70
	.loc	1 108 17                        ; bench_triton_fmha.py:108:17
	s_lshl_b64 s[6:7], s[6:7], 2
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v1, v69
	v_and_b32_e32 v1, 0x1fc, v67
	.loc	1 108 17                        ; bench_triton_fmha.py:108:17
	s_add_u32 s5, s10, s6
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	v_add_u32_e32 v67, 0, v1
	.loc	1 108 17                        ; bench_triton_fmha.py:108:17
	s_addc_u32 s9, s11, s7
	.loc	1 108 34 is_stmt 0              ; bench_triton_fmha.py:108:34
	s_lshl_b64 s[6:7], s[2:3], 2
	.loc	1 109 21 is_stmt 1              ; bench_triton_fmha.py:109:21
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v67, v67
	.loc	1 108 34                        ; bench_triton_fmha.py:108:34
	s_add_u32 s8, s5, s6
	s_addc_u32 s3, s9, s7
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	v_and_b32_e32 v0, 0x80, v0
	s_and_b32 s9, s3, 0xffff
	v_bfrev_b32_e32 v69, 1
	v_cmp_eq_u32_e32 vcc, 0, v0
	.loc	1 110 26                        ; bench_triton_fmha.py:110:26
	s_mul_hi_i32 s3, s0, s2
	s_mul_i32 s2, s0, s2
	.loc	1 95 46                         ; bench_triton_fmha.py:95:46
	v_lshrrev_b32_e32 v66, 3, v140
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	v_cndmask_b32_e32 v0, v69, v1, vcc
	.loc	1 110 26                        ; bench_triton_fmha.py:110:26
	s_lshl_b64 s[2:3], s[2:3], 2
	.loc	1 109 21                        ; bench_triton_fmha.py:109:21
	s_waitcnt lgkmcnt(0)
	buffer_store_dword v67, v0, s[8:11], 0 offen
	.loc	1 110 26                        ; bench_triton_fmha.py:110:26
	s_add_u32 s2, s1, s2
	v_mad_u64_u32 v[0:1], s[0:1], s0, v68, v[66:67]
	s_addc_u32 s3, s4, s3
	v_add_u32_e32 v66, 8, v0
	v_add_u32_e32 v68, 16, v0
	v_add_u32_e32 v70, 24, v0
	v_add_u32_e32 v72, 32, v0
	v_add_u32_e32 v74, 40, v0
	v_add_u32_e32 v76, 48, v0
	v_add_u32_e32 v78, 56, v0
	v_add_u32_e32 v80, 64, v0
	v_add_u32_e32 v82, 0x48, v0
	v_add_u32_e32 v84, 0x50, v0
	v_add_u32_e32 v86, 0x58, v0
	v_add_u32_e32 v88, 0x60, v0
	v_add_u32_e32 v90, 0x68, v0
	v_add_u32_e32 v92, 0x70, v0
	v_add_u32_e32 v94, 0x78, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_ashrrev_i32_e32 v69, 31, v68
	v_ashrrev_i32_e32 v71, 31, v70
	v_ashrrev_i32_e32 v73, 31, v72
	v_ashrrev_i32_e32 v75, 31, v74
	v_ashrrev_i32_e32 v77, 31, v76
	v_ashrrev_i32_e32 v79, 31, v78
	v_ashrrev_i32_e32 v81, 31, v80
	v_ashrrev_i32_e32 v83, 31, v82
	v_ashrrev_i32_e32 v85, 31, v84
	v_ashrrev_i32_e32 v87, 31, v86
	v_ashrrev_i32_e32 v89, 31, v88
	v_ashrrev_i32_e32 v91, 31, v90
	v_ashrrev_i32_e32 v93, 31, v92
	v_ashrrev_i32_e32 v95, 31, v94
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[2:3]
	v_lshl_add_u64 v[68:69], v[68:69], 2, s[2:3]
	v_lshl_add_u64 v[70:71], v[70:71], 2, s[2:3]
	v_lshl_add_u64 v[72:73], v[72:73], 2, s[2:3]
	v_lshl_add_u64 v[74:75], v[74:75], 2, s[2:3]
	v_lshl_add_u64 v[76:77], v[76:77], 2, s[2:3]
	v_lshl_add_u64 v[78:79], v[78:79], 2, s[2:3]
	v_lshl_add_u64 v[80:81], v[80:81], 2, s[2:3]
	v_lshl_add_u64 v[82:83], v[82:83], 2, s[2:3]
	v_lshl_add_u64 v[84:85], v[84:85], 2, s[2:3]
	v_lshl_add_u64 v[86:87], v[86:87], 2, s[2:3]
	v_lshl_add_u64 v[88:89], v[88:89], 2, s[2:3]
	v_lshl_add_u64 v[90:91], v[90:91], 2, s[2:3]
	v_lshl_add_u64 v[92:93], v[92:93], 2, s[2:3]
	v_lshl_add_u64 v[94:95], v[94:95], 2, s[2:3]
	global_store_dwordx4 v[0:1], v[50:53], off
	global_store_dwordx4 v[66:67], v[54:57], off
	global_store_dwordx4 v[68:69], v[58:61], off
	global_store_dwordx4 v[70:71], v[62:65], off
	global_store_dwordx4 v[72:73], v[34:37], off
	global_store_dwordx4 v[74:75], v[38:41], off
	global_store_dwordx4 v[76:77], v[42:45], off
	global_store_dwordx4 v[78:79], v[46:49], off
	global_store_dwordx4 v[80:81], v[18:21], off
	global_store_dwordx4 v[82:83], v[22:25], off
	global_store_dwordx4 v[84:85], v[26:29], off
	global_store_dwordx4 v[86:87], v[30:33], off
	global_store_dwordx4 v[88:89], v[2:5], off
	global_store_dwordx4 v[90:91], v[6:9], off
	global_store_dwordx4 v[92:93], v[10:13], off
	global_store_dwordx4 v[94:95], v[14:17], off
	.loc	1 110 4 is_stmt 0               ; bench_triton_fmha.py:110:4
	s_endpgm
.Ltmp10:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _attn_fwd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 120
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 168
		.amdhsa_next_free_sgpr 52
		.amdhsa_accum_offset 168
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_attn_fwd, .Lfunc_end0-_attn_fwd
	.cfi_endproc
                                        ; -- End function
	.set _attn_fwd.num_vgpr, 168
	.set _attn_fwd.num_agpr, 0
	.set _attn_fwd.numbered_sgpr, 52
	.set _attn_fwd.private_seg_size, 0
	.set _attn_fwd.uses_vcc, 1
	.set _attn_fwd.uses_flat_scratch, 0
	.set _attn_fwd.has_dyn_sized_stack, 0
	.set _attn_fwd.has_recursion, 0
	.set _attn_fwd.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9784
; TotalNumSgprs: 58
; NumVgprs: 168
; NumAgprs: 0
; TotalNumVgprs: 168
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 58
; NumVGPRsForWavesPerEU: 168
; AccumOffset: 168
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 41
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x44 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x1e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	105                             ; DW_AT_call_line
	.byte	60                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"bench_triton_fmha.py"          ; string offset=7
.Linfo_string2:
	.asciz	"/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8" ; string offset=28
.Linfo_string3:
	.asciz	"_attn_fwd"                     ; string offset=76
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         104
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 120
    .max_flat_workgroup_size: 256
    .name:           _attn_fwd
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         _attn_fwd.kd
    .uses_dynamic_stack: false
    .vgpr_count:     168
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:

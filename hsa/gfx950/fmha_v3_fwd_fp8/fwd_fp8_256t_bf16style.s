// FP8 Flash Attention - 256T following BF16 pattern exactly
// Direct port from fwd_hd128_bf16.s
//
// BF16 patterns preserved:
// - LDS swizzle: m0 = 0x8200 + 0x408 * wave_id
// - Per-wave Q offset: wave_id * Q_row_stride * 2
// - buffer_load_dwordx4 ... offen lds (direct-to-LDS)
// - 8 MFMA per K-slice (HD=128 / 16 = 8)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter19fwd_fp8_256t_bf16stE
.p2align 8
.type _ZN5aiter19fwd_fp8_256t_bf16stE,@function

_ZN5aiter19fwd_fp8_256t_bf16stE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // KERNEL ARGS (simplified vs BF16)
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O ptr
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q ptr
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K ptr
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V ptr
    s_load_dword s24, s[0:1], 0x20         // seq_len
    s_load_dword s25, s[0:1], 0x24         // Q_row_stride (bytes)
    
    // ========================================================================
    // THREAD SETUP (same as BF16 lines 44-54)
    // ========================================================================
    // v0 = tid (0-255)
    // wave_id = tid >> 6
    // lane_id = tid & 63
    v_lshrrev_b32_e32 v3, 6, v0           // wave_id (0-3)
    v_and_b32_e32 v0, 63, v0              // lane_id (0-63)
    v_readfirstlane_b32 s5, v3            // s5 = wave_id (SGPR)
    
    s_waitcnt lgkmcnt(0)
    
    // Number of K-tiles
    s_add_i32 s26, s24, 31
    s_lshr_b32 s26, s26, 5                // num_tiles = (seq_len + 31) / 32
    
    // ========================================================================
    // BUFFER DESCRIPTORS
    // ========================================================================
    // Q buffer desc
    s_mov_b32 s10, -1                     // unlimited size
    s_mov_b32 s11, 0x20000                // num_format
    
    // K buffer desc
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    
    // V buffer desc
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000
    
    // ========================================================================
    // Q LOAD ADDRESS CALCULATION (from BF16 lines 162-175)
    // Each wave loads different Q rows
    // ========================================================================
    
    // Per-thread offset within wave
    v_lshrrev_b32_e32 v12, 3, v0          // v0 / 8
    v_and_b32_e32 v13, 1, v12             // (v0/8) & 1
    v_mul_i32_i24_e32 v13, s25, v13       // * Q_row_stride
    v_lshrrev_b32_e32 v14, 1, v12         // (v0/8) / 2
    v_mul_i32_i24_e32 v14, s25, v14       // * Q_row_stride
    v_mul_i32_i24_e32 v14, 32, v14        // * 32
    v_and_b32_e32 v12, 7, v0              // v0 & 7
    v_lshlrev_b32_e32 v12, 4, v12         // * 16 (bytes per load chunk)
    
    // Per-wave offset
    s_mul_i32 s40, s5, s25                // wave_id * Q_row_stride
    s_mul_i32 s40, 2, s40                 // * 2 (2 rows per wave section)
    
    // Combine
    v_add_u32_e32 v4, s40, v12
    v_add_u32_e32 v4, v13, v4
    v_add_u32_e32 v4, v14, v4
    
    // Secondary addresses for Q loads
    s_mul_i32 s40, 16, s25                // 16 * Q_row_stride
    v_add_u32_e32 v5, s40, v4
    v_add_u32_e32 v6, 0x80, v4            // +128 bytes
    v_add_u32_e32 v7, s40, v6
    
    // ========================================================================
    // LDS SWIZZLE SETUP (from BF16 lines 180-182)
    // m0 = 0x8200 + 0x408 * wave_id
    // ========================================================================
    s_mul_i32 s63, 0x408, s5
    s_add_u32 s63, 0x8200, s63
    s_mov_b32 m0, s63
    
    // ========================================================================
    // LOAD Q TO LDS (cooperative, all 256 threads)
    // ========================================================================
    
    // Q row stride in LDS
    s_mul_i32 s40, s25, 0x80              // row_stride * 128
    
    buffer_load_dwordx4 v4, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0              // BF16 pattern: stride 0x2040
    
    buffer_load_dwordx4 v5, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    
    buffer_load_dwordx4 v6, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    
    buffer_load_dwordx4 v7, s[8:11], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    
    // ========================================================================
    // INITIALIZE OUTPUT ACCUMULATORS (same VGPR layout as BF16)
    // v[96:159] = 64 VGPRs for O accumulation (4 output tiles × 16 values)
    // ========================================================================
    
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
    
    // Online softmax state
    s_mov_b32 s29, 0x3fb8aa3b             // log2(e)
    v_mov_b32_e32 v27, 0xff800000         // -inf (running_max init)
    v_mov_b32_e32 v18, 0                  // running_sum
    
    // K-tile loop setup
    s_mov_b32 s34, 0                      // k_tile_idx = 0
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
K_TILE_LOOP:
    
    // --------------------------------------------------------------------
    // LOAD K TO LDS (similar to BF16)
    // --------------------------------------------------------------------
    
    // K address: K_ptr + k_tile_idx * 32 * 128
    s_lshl_b32 s35, s34, 12               // k_tile_idx * 4096
    
    // Reset m0 for K load
    s_mov_b32 m0, s63
    
    // Each thread loads 16 bytes of K
    v_lshrrev_b32_e32 v12, 3, v0
    v_and_b32_e32 v12, 7, v0
    v_lshlrev_b32_e32 v12, 4, v12         // lane * 16
    v_add_u32_e32 v12, s35, v12           // + tile offset
    
    buffer_load_dwordx4 v12, s[12:15], 0 offen lds
    s_add_u32 m0, 0x2040, m0
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    
    // --------------------------------------------------------------------
    // QK MFMA: S = Q @ K^T (8 MFMAs for HD=128)
    // Following BF16 pattern with FP8 MFMA
    // --------------------------------------------------------------------
    
    // LDS addresses for Q and K (using swizzled layout)
    // Q: starts at 0x8200 region
    // K: starts at 0x0 region (separate from Q)
    
    // For simplicity, use ds_read for now (will optimize to match BF16 later)
    // Q addr: mfma_row_mapping * 128 + k_offset
    // K addr: mfma_row_mapping * 128 + k_offset
    
    // MFMA row mapping (same as before)
    v_and_b32_e32 v1, 3, v0
    v_lshrrev_b32_e32 v2, 3, v0
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v80, v1, v2
    v_lshrrev_b32_e32 v4, 2, v0
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v80, v80, v4            // mfma_row
    
    // k_base: 0 for lanes 0-31, 8 for lanes 32-63
    v_mov_b32_e32 v81, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v0, 32
    v_cndmask_b32_e32 v81, v81, v4, vcc   // k_base
    
    // Q LDS read address
    v_lshlrev_b32_e32 v70, 7, v80         // mfma_row * 128
    v_add_u32_e32 v70, v70, v81           // + k_base
    v_add_u32_e32 v70, 0x8200, v70        // + Q region base
    
    // K LDS read address (K is at 0x0)
    v_lshlrev_b32_e32 v71, 7, v80
    v_add_u32_e32 v71, v71, v81
    
    // Clear S accumulators
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
    
    // 8 MFMAs for HD=128 (K dimension loop)
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v71    // K addr
        ds_read_b64 v[60:61], v72
        v_add_u32_e32 v73, \k_off, v70    // Q addr
        ds_read_b64 v[62:63], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[60:61], v[62:63], v[32:47]
        s_nop 7
    .endr
    
    s_nop 15
    
    // S is now in v[32:47]
    
    // --------------------------------------------------------------------
    // SOFTMAX (simplified for now)
    // Scale by 1/sqrt(d), find max, subtract max, exp
    // --------------------------------------------------------------------
    
    // Scale by log2(e)/sqrt(128) = 0.101...
    s_mov_b32 s40, 0x3e028f5c             // log2(e)/sqrt(128)
    v_mul_f32_e32 v32, s40, v32
    v_mul_f32_e32 v33, s40, v33
    v_mul_f32_e32 v34, s40, v34
    v_mul_f32_e32 v35, s40, v35
    v_mul_f32_e32 v36, s40, v36
    v_mul_f32_e32 v37, s40, v37
    v_mul_f32_e32 v38, s40, v38
    v_mul_f32_e32 v39, s40, v39
    v_mul_f32_e32 v40, s40, v40
    v_mul_f32_e32 v41, s40, v41
    v_mul_f32_e32 v42, s40, v42
    v_mul_f32_e32 v43, s40, v43
    v_mul_f32_e32 v44, s40, v44
    v_mul_f32_e32 v45, s40, v45
    v_mul_f32_e32 v46, s40, v46
    v_mul_f32_e32 v47, s40, v47
    
    // Find max (simplified - just first element for now)
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
    
    // Update running max
    v_max_f32_e32 v27, v27, v21
    
    // Subtract max and exp (simplified)
    v_sub_f32_e32 v32, v32, v27
    v_sub_f32_e32 v33, v33, v27
    v_sub_f32_e32 v34, v34, v27
    v_sub_f32_e32 v35, v35, v27
    v_sub_f32_e32 v36, v36, v27
    v_sub_f32_e32 v37, v37, v27
    v_sub_f32_e32 v38, v38, v27
    v_sub_f32_e32 v39, v39, v27
    v_sub_f32_e32 v40, v40, v27
    v_sub_f32_e32 v41, v41, v27
    v_sub_f32_e32 v42, v42, v27
    v_sub_f32_e32 v43, v43, v27
    v_sub_f32_e32 v44, v44, v27
    v_sub_f32_e32 v45, v45, v27
    v_sub_f32_e32 v46, v46, v27
    v_sub_f32_e32 v47, v47, v27
    
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
    
    // Convert P to FP8 for PV MFMA
    v_cvt_pk_fp8_f32 v48, v32, v33
    v_cvt_pk_fp8_f32 v49, v34, v35
    v_cvt_pk_fp8_f32 v50, v36, v37
    v_cvt_pk_fp8_f32 v51, v38, v39
    
    // --------------------------------------------------------------------
    // LOAD V TO LDS
    // --------------------------------------------------------------------
    
    s_mov_b32 m0, s63
    buffer_load_dwordx4 v12, s[16:19], s35 offen lds
    s_add_u32 m0, 0x2040, m0
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    
    // --------------------------------------------------------------------
    // PV MFMA: O += P @ V
    // --------------------------------------------------------------------
    
    // V LDS address
    v_lshlrev_b32_e32 v74, 7, v80
    v_add_u32_e32 v74, v74, v81
    
    // Read V
    ds_read_b64 v[60:61], v74
    s_waitcnt lgkmcnt(0)
    
    // Pack P to AGPR
    v_accvgpr_write_b32 a0, v48
    v_accvgpr_write_b32 a1, v49
    
    // PV MFMA
    v_mfma_f32_32x32x16_fp8_fp8 v[96:111], a[0:1], v[60:61], v[96:111]
    s_nop 15
    
    // --------------------------------------------------------------------
    // LOOP INCREMENT
    // --------------------------------------------------------------------
    
    s_add_i32 s34, s34, 1
    s_cmp_lt_i32 s34, s26
    s_cbranch_scc1 K_TILE_LOOP
    
    // ========================================================================
    // OUTPUT (all 4 waves write their results)
    // ========================================================================
    
    // O offset: wave_id * 32 * 128 (each wave handles 32 output rows)
    s_mul_i32 s40, s5, 32
    s_mul_i32 s40, s40, 128
    s_mul_i32 s40, s40, 4                 // * sizeof(float)
    
    v_mov_b32_e32 v10, s4
    v_mov_b32_e32 v11, s5
    
    // Per-lane offset
    v_lshlrev_b32_e32 v12, 6, v0          // lane * 64 bytes
    v_add_u32_e32 v12, s40, v12
    v_add_co_u32_e32 v10, vcc, v12, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_store_dwordx4 v[10:11], v[96:99]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[100:103]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[104:107]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[108:111]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter19fwd_fp8_256t_bf16stE
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 176
    .amdhsa_next_free_sgpr 68
    .amdhsa_accum_offset 176
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter19fwd_fp8_256t_bf16stE
    .symbol: _ZN5aiter19fwd_fp8_256t_bf16stE.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 68
    .vgpr_count: 176
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
      - {.name: Q_row_stride, .size: 4, .offset: 36, .value_kind: by_value}
...
.end_amdgpu_metadata

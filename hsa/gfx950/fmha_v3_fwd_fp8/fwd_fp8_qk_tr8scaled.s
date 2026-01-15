// FP8 QK with TR8 reads using BF16-style scaled offsets
// Key insight: TR16 offsets scaled by 0.5 for TR8 (8-bit vs 16-bit)
// BF16 TR16: 0, 512, 64, 576, ... -> TR8: 0, 256, 32, 288, ...
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl fwd_fp8_qk_tr8scaled
.p2align 8
.type fwd_fp8_qk_tr8scaled,@function

fwd_fp8_qk_tr8scaled:
    s_mov_b64 exec, -1
    
    // Args: O, Q, K, num_k_tiles
    s_load_dwordx2 s[20:21], s[0:1], 0x00  // O
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K
    s_load_dword s6, s[0:1], 0x18          // num_k_tiles
    
    v_mov_b32_e32 v0, v0
    v_lshrrev_b32_e32 v1, 6, v0            // wave_id
    v_and_b32_e32 v2, 63, v0               // lane_id
    v_readfirstlane_b32 s5, v1
    
    s_waitcnt lgkmcnt(0)
    
    // Buffer descriptors
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s22, -1
    s_mov_b32 s23, 0x20000
    
    // Load Q to LDS with BF16-style swizzle
    s_mul_i32 s63, 0x408, s5
    s_add_u32 s63, 0x8200, s63
    s_mov_b32 m0, s63
    v_lshlrev_b32_e32 v4, 4, v0            // tid * 16
    s_mov_b32 s40, 0
    buffer_load_dwordx4 v4, s[8:11], s40 offen lds
    s_add_u32 m0, 0x1000, m0
    s_mov_b32 s40, 0x1000
    buffer_load_dwordx4 v4, s[8:11], s40 offen lds
    s_add_u32 m0, 0x1000, m0
    s_mov_b32 s40, 0x2000
    buffer_load_dwordx4 v4, s[8:11], s40 offen lds
    s_add_u32 m0, 0x1000, m0
    s_mov_b32 s40, 0x3000
    buffer_load_dwordx4 v4, s[8:11], s40 offen lds
    s_waitcnt vmcnt(0)
    s_barrier
    
    // TR8 base address - MATCH BF16's v10 formula but scaled for 8-bit
    // BF16 v10 formula:
    //   v10 = 0x8200 + (wave*0x440) + (tid&3)*8 + ((tid>>4)&1)*32 + (tid>>5)*256
    // For TR8 with 8-bit elements, scale the offsets by 0.5:
    //   base = 0x8200 + (wave*0x220) + (tid&3)*4 + ((tid>>4)&1)*16 + (tid>>5)*128
    // But we're using same m0 swizzle on write, so use same read base
    
    // Simpler approach: use BF16's exact address formula
    v_and_b32_e32 v3, 3, v0                // tid & 3
    v_lshlrev_b32_e32 v3, 3, v3            // * 8 (same as BF16)
    v_lshrrev_b32_e32 v4, 4, v0            // tid >> 4
    v_and_b32_e32 v5, 1, v4                // (tid>>4) & 1
    v_lshlrev_b32_e32 v5, 5, v5            // * 32
    v_add_u32_e32 v3, v3, v5
    v_lshrrev_b32_e32 v5, 1, v4            // tid >> 5
    v_lshlrev_b32_e32 v5, 8, v5            // * 256
    v_add_u32_e32 v3, v3, v5
    v_add_u32_e32 v3, 0x8200, v3           // + base
    
    // K TR8 addresses (same formula, no base offset for buffer 0)
    v_and_b32_e32 v70, 3, v0               // tid & 3
    v_lshlrev_b32_e32 v70, 3, v70          // * 8
    v_lshrrev_b32_e32 v4, 4, v0            // tid >> 4
    v_and_b32_e32 v5, 1, v4                // (tid>>4) & 1
    v_lshlrev_b32_e32 v5, 5, v5            // * 32
    v_add_u32_e32 v70, v70, v5
    v_lshrrev_b32_e32 v5, 1, v4            // tid >> 5
    v_lshlrev_b32_e32 v5, 8, v5            // * 256
    v_add_u32_e32 v70, v70, v5             // Buffer 0 base
    v_add_u32_e32 v71, 4096, v70           // Buffer 1 base
    
    // Read Q with TR8 using SCALED TR16 offsets
    // TR16 offsets: 0, 512, 64, 576, 2176, 2688, 2240, 2752
    // TR8 scaled:   0, 256, 32, 288, 1088, 1344, 1120, 1376
    ds_read_b64_tr_b8 v[100:101], v3
    ds_read_b64_tr_b8 v[102:103], v3 offset:256
    ds_read_b64_tr_b8 v[104:105], v3 offset:32
    ds_read_b64_tr_b8 v[106:107], v3 offset:288
    ds_read_b64_tr_b8 v[108:109], v3 offset:1088
    ds_read_b64_tr_b8 v[110:111], v3 offset:1344
    ds_read_b64_tr_b8 v[112:113], v3 offset:1120
    ds_read_b64_tr_b8 v[114:115], v3 offset:1376
    s_waitcnt lgkmcnt(0)
    
    // K load offsets
    v_lshlrev_b32_e32 v5, 4, v0
    v_add_u32_e32 v6, 4096, v5
    
    // Init accumulators
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
    
    // Prologue: Load K[0]
    s_mul_i32 s63, 0x408, s5
    s_mov_b32 m0, s63
    s_mov_b32 s30, 0
    buffer_load_dwordx4 v5, s[12:15], s30 offen lds
    s_waitcnt vmcnt(0)
    s_barrier
    
    s_add_u32 s30, s30, 4096
    s_sub_u32 s6, s6, 1
    s_cmp_eq_u32 s6, 0
    s_cbranch_scc1 LAST_TILE

EVEN_ITER:
    s_add_u32 s63, 4096, s63
    s_mov_b32 m0, s63
    buffer_load_dwordx4 v6, s[12:15], s30 offen lds
    
    // TR8 reads with scaled offsets
    ds_read_b64_tr_b8 v[116:117], v70
    ds_read_b64_tr_b8 v[118:119], v70 offset:256
    ds_read_b64_tr_b8 v[120:121], v70 offset:32
    ds_read_b64_tr_b8 v[122:123], v70 offset:288
    ds_read_b64_tr_b8 v[124:125], v70 offset:1088
    ds_read_b64_tr_b8 v[126:127], v70 offset:1344
    ds_read_b64_tr_b8 v[128:129], v70 offset:1120
    ds_read_b64_tr_b8 v[130:131], v70 offset:1376
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[100:101], v[116:117], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[102:103], v[118:119], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[104:105], v[120:121], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[106:107], v[122:123], v[32:47]
    s_waitcnt vmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[108:109], v[124:125], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[110:111], v[126:127], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[112:113], v[128:129], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[114:115], v[130:131], v[32:47]
    s_barrier
    
    s_add_u32 s30, s30, 4096
    s_sub_u32 s6, s6, 1
    s_cmp_eq_u32 s6, 0
    s_cbranch_scc1 LAST_TILE_BUF1

ODD_ITER:
    s_sub_u32 s63, s63, 4096
    s_mov_b32 m0, s63
    buffer_load_dwordx4 v5, s[12:15], s30 offen lds
    
    ds_read_b64_tr_b8 v[116:117], v71
    ds_read_b64_tr_b8 v[118:119], v71 offset:256
    ds_read_b64_tr_b8 v[120:121], v71 offset:32
    ds_read_b64_tr_b8 v[122:123], v71 offset:288
    ds_read_b64_tr_b8 v[124:125], v71 offset:1088
    ds_read_b64_tr_b8 v[126:127], v71 offset:1344
    ds_read_b64_tr_b8 v[128:129], v71 offset:1120
    ds_read_b64_tr_b8 v[130:131], v71 offset:1376
    s_waitcnt lgkmcnt(0)
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[100:101], v[116:117], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[102:103], v[118:119], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[104:105], v[120:121], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[106:107], v[122:123], v[32:47]
    s_waitcnt vmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[108:109], v[124:125], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[110:111], v[126:127], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[112:113], v[128:129], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[114:115], v[130:131], v[32:47]
    s_barrier
    
    s_add_u32 s30, s30, 4096
    s_sub_u32 s6, s6, 1
    s_cmp_lg_u32 s6, 0
    s_cbranch_scc1 EVEN_ITER

LAST_TILE:
    ds_read_b64_tr_b8 v[116:117], v70
    ds_read_b64_tr_b8 v[118:119], v70 offset:256
    ds_read_b64_tr_b8 v[120:121], v70 offset:32
    ds_read_b64_tr_b8 v[122:123], v70 offset:288
    ds_read_b64_tr_b8 v[124:125], v70 offset:1088
    ds_read_b64_tr_b8 v[126:127], v70 offset:1344
    ds_read_b64_tr_b8 v[128:129], v70 offset:1120
    ds_read_b64_tr_b8 v[130:131], v70 offset:1376
    s_waitcnt lgkmcnt(0)
    s_branch DO_LAST_MFMA

LAST_TILE_BUF1:
    ds_read_b64_tr_b8 v[116:117], v71
    ds_read_b64_tr_b8 v[118:119], v71 offset:256
    ds_read_b64_tr_b8 v[120:121], v71 offset:32
    ds_read_b64_tr_b8 v[122:123], v71 offset:288
    ds_read_b64_tr_b8 v[124:125], v71 offset:1088
    ds_read_b64_tr_b8 v[126:127], v71 offset:1344
    ds_read_b64_tr_b8 v[128:129], v71 offset:1120
    ds_read_b64_tr_b8 v[130:131], v71 offset:1376
    s_waitcnt lgkmcnt(0)

DO_LAST_MFMA:
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[100:101], v[116:117], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[102:103], v[118:119], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[104:105], v[120:121], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[106:107], v[122:123], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[108:109], v[124:125], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[110:111], v[126:127], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[112:113], v[128:129], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], v[114:115], v[130:131], v[32:47]
    
    s_nop 15
    s_nop 7
    
    // Output
    v_mov_b32_e32 v10, s5
    v_lshlrev_b32_e32 v10, 6, v10
    v_add_u32_e32 v10, v2, v10
    v_lshlrev_b32_e32 v10, 6, v10
    
    buffer_store_dwordx4 v[32:35], v10, s[20:23], 0 offen
    v_add_u32_e32 v10, 16, v10
    buffer_store_dwordx4 v[36:39], v10, s[20:23], 0 offen
    v_add_u32_e32 v10, 16, v10
    buffer_store_dwordx4 v[40:43], v10, s[20:23], 0 offen
    v_add_u32_e32 v10, 16, v10
    buffer_store_dwordx4 v[44:47], v10, s[20:23], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel fwd_fp8_qk_tr8scaled
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 132
    .amdhsa_next_free_sgpr 68
    .amdhsa_accum_offset 132
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
  - .name: fwd_fp8_qk_tr8scaled
    .symbol: fwd_fp8_qk_tr8scaled.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 68
    .vgpr_count: 132
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: num_k_tiles, .size: 4, .offset: 24, .value_kind: by_value}
...
.end_amdgpu_metadata

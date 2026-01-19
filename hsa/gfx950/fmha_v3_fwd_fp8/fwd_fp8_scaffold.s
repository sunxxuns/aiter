// FP8 QK+PV scaffold kernel (no softmax) for perf validation
// - Focus: LDS layout + bulk loads/stores + MFMA throughput
// - NOT numerically correct (no softmax, P reuse)
// Grid: (num_q_blocks, batch * heads, 1)
// Each block: 256 threads (4 waves), processes 128 Q rows

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.set Q_LDS, 0                 // Q tile 0
.set Q_LDS1, 16896            // Q tile 1 (128×132)
.set K_LDS0, 33792            // 32×128 (row-major)
.set V_LDS0, 37888            // 32×128 (row-major)
.set K_LDS1, 41984            // ping-pong
.set V_LDS1, 46080            // ping-pong
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
    s_barrier

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

    buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 offen
    v_add_u32_e32 v3, 16, v2
    buffer_load_dwordx4 v[8:11], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 32, v2
    buffer_load_dwordx4 v[12:15], v3, s[8:11], 0 offen
    v_add_u32_e32 v3, 48, v2
    buffer_load_dwordx4 v[16:19], v3, s[8:11], 0 offen
    s_waitcnt vmcnt(0)

    ds_write_b128 v20, v[4:7]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[8:11]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[12:15]
    v_add_u32_e32 v20, 16, v20
    ds_write_b128 v20, v[16:19]

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
    s_mov_b32 s30, 0                     // tile_idx
    s_mov_b32 s31, 0                     // K/V tile soffset

    // Preload tile 0 into buffer 0 (all threads, masked vaddr)
    v_mov_b32_e32 v13, v61
    s_mov_b32 m0, K_LDS0
    buffer_load_dwordx4 v13, s[12:15], s31 offen lds
    s_mov_b32 m0, V_LDS0
    buffer_load_dwordx4 v13, s[16:19], s31 offen lds
    s_waitcnt vmcnt(0)

K_LOOP:
    s_cmp_ge_u32 s30, s20
    s_cbranch_scc1 K_LOOP_END

    // Select ping-pong LDS base for current tile
    v_mov_b32_e32 v28, s30
    v_and_b32_e32 v28, 1, v28
    v_cmp_eq_u32_e64 vcc, v28, 0
    v_mov_b32_e32 v29, s40                      // K_LDS0
    v_mov_b32_e32 v31, s41                      // K_LDS1
    v_cndmask_b32_e64 v29, v31, v29, vcc        // K base
    v_mov_b32_e32 v56, s42                      // V_LDS0
    v_mov_b32_e32 v57, s43                      // V_LDS1
    v_cndmask_b32_e64 v56, v57, v56, vcc        // V base

    // Compute Q/K read addresses for this tile
    v_add_u32_e32 v26, v29, v18                 // K base + row
    v_add_u32_e32 v27, v26, v11                 // K addr1
    v_add_u32_e32 v28, v26, v12                 // K addr2

    // Prefetch next tile into other buffer (if any)
    s_add_u32 s34, s30, 1
    s_cmp_ge_u32 s34, s20
    s_cbranch_scc1 PREFETCH_DONE

    s_and_b32 s35, s30, 1
    s_cmp_eq_u32 s35, 0
    s_cselect_b32 s36, s41, s40          // other K base
    s_cselect_b32 s37, s43, s42          // other V base
    s_add_u32 s38, s31, 4096             // next tile offset

    v_mov_b32_e32 v13, v61
    s_mov_b32 m0, s36
    buffer_load_dwordx4 v13, s[12:15], s38 offen lds
    s_mov_b32 m0, s37
    buffer_load_dwordx4 v13, s[16:19], s38 offen lds

PREFETCH_DONE:

    // QK MFMA (K=64 × 2)
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

    // V tile already prefetched into LDS (ping-pong)

    // PV MFMA using TR8 V reads (K=16 MFMA, scaffold only)
    // Use bank-conflict-free TR8 base: (lane%32)*128 + (lane/32)*32
    v_and_b32_e32 v2, 31, v10
    v_lshrrev_b32_e32 v3, 5, v10
    v_lshlrev_b32_e32 v4, 7, v2
    v_lshlrev_b32_e32 v5, 5, v3
    v_add_u32_e32 v6, v4, v5

    .macro READ_V_TR8_4COL k_row_offset
        v_add_u32_e32 v5, v56, v6
        v_add_u32_e32 v5, \k_row_offset, v5
        ds_read_b64_tr_b8 v[30:31], v5 offset:0
        ds_read_b64_tr_b8 v[32:33], v5 offset:32
        ds_read_b64_tr_b8 v[34:35], v5 offset:64
        ds_read_b64_tr_b8 v[36:37], v5 offset:96
    .endm

    // K-pass 0 (k=0..15) + K-pass 1 (k=16..31), batch V reads
    READ_V_TR8_4COL 0
    // second pass into v38:45
    v_lshlrev_b32_e32 v5, 10, v3
    v_add_u32_e32 v5, v5, v2
    v_add_u32_e32 v5, v56, v5
    v_add_u32_e32 v5, 2048, v5
    ds_read_b64_tr_b8 v[38:39], v5 offset:0
    ds_read_b64_tr_b8 v[40:41], v5 offset:32
    ds_read_b64_tr_b8 v[42:43], v5 offset:64
    ds_read_b64_tr_b8 v[44:45], v5 offset:96

    // Convert P (v[32:47]) to FP8 packed (v48-v55)
    v_mov_b32_e32 v59, 0x05040100
    v_cvt_pk_fp8_f32 v48, v32, v33
    v_cvt_pk_fp8_f32 v49, v34, v35
    v_perm_b32 v48, v48, v49, v59

    v_cvt_pk_fp8_f32 v49, v36, v37
    v_cvt_pk_fp8_f32 v50, v38, v39
    v_perm_b32 v49, v49, v50, v59
    s_waitcnt lgkmcnt(0)

    v_accvgpr_write_b32 a0, v48
    v_accvgpr_write_b32 a1, v49
    s_nop 1
    // K-pass 0
    v_mfma_f32_32x32x16_fp8_fp8 v[64:79], a[0:1], v[30:31], v[64:79]
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[32:33], v[80:95]
    v_mfma_f32_32x32x16_fp8_fp8 v[96:111], a[0:1], v[34:35], v[96:111]
    v_mfma_f32_32x32x16_fp8_fp8 v[112:127], a[0:1], v[36:37], v[112:127]
    // K-pass 1 (reuse same P)
    v_mfma_f32_32x32x16_fp8_fp8 v[64:79], a[0:1], v[38:39], v[64:79]
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[40:41], v[80:95]
    v_mfma_f32_32x32x16_fp8_fp8 v[96:111], a[0:1], v[42:43], v[96:111]
    v_mfma_f32_32x32x16_fp8_fp8 v[112:127], a[0:1], v[44:45], v[112:127]

    // Advance to next K/V tile
    s_add_u32 s31, s31, 4096
    s_add_u32 s30, s30, 1
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

// FP8 Flash Attention with K-loop and Pitch-136 LDS layout
// 256 threads (4 waves), buffer_load...lds, full attention
// O = softmax(Q @ K^T / sqrt(d)) @ V
//
// Architecture matches BF16 reference:
// - Load Q once, keep in LDS
// - K-loop: iterate over K/V tiles
// - Online softmax: track running max/sum
// - PV MFMA: accumulate output

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136                     // Row pitch for zero bank conflicts
.set Q_LDS, 0                       // Q at LDS[0], 32*136=4352 bytes
.set K_LDS, 4352                    // K after Q
.set V_LDS, 8704                    // V after K
.set P_LDS, 13056                   // P after V (32*32*4=4096 bytes)
.set LDS_SIZE, 20480                // Total LDS
.set SCALE, 0x3e028f5c              // log2(e) / sqrt(128)
.set NEG_INF, 0xff800000            // -inf for softmax init

.text
.globl _ZN5aiter16fwd_fp8_kloop_p136E
.p2align 8
.type _ZN5aiter16fwd_fp8_kloop_p136E,@function

_ZN5aiter16fwd_fp8_kloop_p136E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // LOAD KERNEL ARGUMENTS
    // ========================================================================
    // Args: O_ptr, Q_ptr, K_ptr, V_ptr, seq_len, head_dim (128)
    s_load_dwordx2 s[4:5], s[0:1], 0      // O_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr
    s_load_dwordx2 s[16:17], s[0:1], 24   // V_ptr
    s_load_dword s20, s[0:1], 32          // seq_len
    s_waitcnt lgkmcnt(0)
    
    // Thread IDs
    v_and_b32_e32 v60, 63, v0             // lane_id
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id (0-3)
    v_and_b32_e32 v62, 255, v0            // tid (0-255)
    
    // ========================================================================
    // SET UP BUFFER DESCRIPTORS
    // ========================================================================
    // Q buffer: s[8:11]
    s_mov_b32 s10, -1                     // size = max
    s_mov_b32 s11, 0x20000                // raw buffer format
    
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
    // COMPUTE THREAD OFFSETS FOR LOADING
    // ========================================================================
    // Each thread loads 16 bytes (dwordx4)
    // 256 threads × 16 bytes = 4096 bytes = 32 rows × 128 cols
    // Row = tid / 8, col_chunk = tid % 8
    
    v_lshrrev_b32_e32 v1, 3, v62          // row = tid / 8
    v_and_b32_e32 v2, 7, v62              // col_chunk = tid % 8
    
    // Global offset: row * 128 + col_chunk * 16
    v_lshlrev_b32_e32 v3, 7, v1           // row * 128
    v_lshlrev_b32_e32 v4, 4, v2           // col_chunk * 16
    v_add_u32_e32 v50, v3, v4             // v50 = global load offset
    
    // LDS offset with pitch-136: row * 136 + col_chunk * 16
    v_mov_b32_e32 v5, PITCH
    v_mul_lo_u32 v51, v1, v5              // row * 136
    v_add_u32_e32 v51, v51, v4            // + col_chunk * 16
    // v51 = LDS offset (relative to base)
    
    // ========================================================================
    // LOAD Q TO LDS (only once, keep for all K iterations)
    // ========================================================================
    v_add_u32_e32 v52, Q_LDS, v51         // v52 = Q LDS addr
    s_mov_b32 m0, 0                        // m0 not used for offen+lds? 
    
    // Use buffer_load with offen (offset in VGPR)
    // The lds modifier writes to LDS at address = m0 + (vgpr offset)?
    // Actually for buffer_load...lds, the LDS address comes from the VGPR
    // Let me use the simpler approach: load to VGPR then ds_write
    
    buffer_load_dwordx4 v[20:23], v50, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    ds_write_b128 v52, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE MFMA ADDRESSING
    // ========================================================================
    // MFMA row mapping for 32x32x16
    v_and_b32_e32 v1, 3, v60              // lane & 3
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2               // (lane >> 3) & 3
    v_lshlrev_b32_e32 v2, 2, v2           // * 4
    v_add_u32_e32 v3, v1, v2              // row_in_16
    
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4               // (lane >> 2) & 1
    v_lshlrev_b32_e32 v4, 4, v4           // * 16
    v_add_u32_e32 v63, v3, v4             // v63 = mfma_row (0-31)
    
    // k_half: lanes 0-31 read k=0-7, lanes 32-63 read k=8-15
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc      // v64 = k_half
    
    // Q read address: Q_LDS + mfma_row * PITCH + k_half
    v_mov_b32_e32 v1, PITCH
    v_mul_lo_u32 v70, v63, v1             // mfma_row * PITCH
    v_add_u32_e32 v70, v70, v64           // + k_half
    v_add_u32_e32 v70, Q_LDS, v70         // v70 = Q read base
    
    // ========================================================================
    // INITIALIZE SOFTMAX STATE AND OUTPUT ACCUMULATORS
    // ========================================================================
    s_mov_b32 s2, SCALE
    s_mov_b32 s3, NEG_INF
    
    v_mov_b32_e32 v65, s3                 // v65 = row_max = -inf
    v_mov_b32_e32 v66, 0                  // v66 = row_sum = 0
    
    // O accumulators (4 HD tiles × 16 regs = 64 regs, but start with 1 tile)
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // K-LOOP: ITERATE OVER K/V TILES
    // ========================================================================
    // For now, single K tile (seq_len=32) to verify correctness
    // TODO: Add loop for seq_len > 32
    
    // Load K tile to LDS
    v_add_u32_e32 v53, K_LDS, v51         // K LDS addr
    buffer_load_dwordx4 v[24:27], v50, s[12:15], 0 offen
    s_waitcnt vmcnt(0)
    ds_write_b128 v53, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // QK MFMA: S = Q @ K^T
    // ========================================================================
    // K read address: K_LDS + mfma_row * PITCH + k_half
    // v70 = Q_LDS + offset, so K = K_LDS + offset = v70 - Q_LDS + K_LDS
    v_add_u32_e32 v71, K_LDS - Q_LDS, v70 // v71 = K read base
    
    // Clear S accumulators
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // 8 MFMA iterations for HD=128
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v70
        ds_read_b64 v[30:31], v72
        
        v_add_u32_e32 v73, \k_off, v71
        ds_read_b64 v[32:33], v73
        
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    s_nop 15
    
    // S is now in v[0:15] (16 F32 values per lane)
    
    // ========================================================================
    // ONLINE SOFTMAX: UPDATE MAX AND COMPUTE EXP
    // ========================================================================
    
    // Find row max across this tile's 16 values
    v_max_f32_e32 v20, v0, v1
    .irp i, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_max_f32_e32 v20, v20, v\i
    .endr
    
    // Cross-lane max (lanes 0-31 and 32-63 share a row)
    v_mov_b32_e32 v21, v20
    s_nop 1
    v_permlane32_swap_b32_e32 v21, v20
    v_max_f32_e32 v20, v20, v21           // v20 = tile_max
    
    // Update running max
    v_max_f32_e32 v67, v65, v20           // v67 = new_max
    
    // Correction factor for old sum: exp(old_max - new_max)
    v_sub_f32_e32 v21, v65, v67           // old_max - new_max
    v_mul_f32_e32 v21, s2, v21            // * scale
    v_exp_f32_e32 v68, v21                // v68 = correction
    s_nop 7
    s_nop 7
    s_nop 7
    
    // Correct old sum
    v_mul_f32_e32 v66, v66, v68           // row_sum *= correction
    
    // Compute exp((S - new_max) * scale) for this tile
    v_mul_f32_e32 v21, s2, v67            // scaled_max = new_max * scale
    
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_fma_f32 v\i, v\i, s2, -v21      // (S - max) * scale
        v_exp_f32_e32 v\i, v\i
    .endr
    s_nop 7
    s_nop 7
    s_nop 7
    
    // Update running sum
    v_add_f32_e32 v22, v0, v1
    .irp i, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_add_f32_e32 v22, v22, v\i
    .endr
    
    // Cross-lane sum
    v_mov_b32_e32 v23, v22
    s_nop 1
    v_permlane32_swap_b32_e32 v23, v22
    v_add_f32_e32 v22, v22, v23           // v22 = tile_sum
    
    v_add_f32_e32 v66, v66, v22           // row_sum += tile_sum
    v_mov_b32_e32 v65, v67                // row_max = new_max
    
    // Correct old O accumulator
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v\i, v68
    .endr
    
    // P (unnormalized) is now in v[0:15]
    
    // ========================================================================
    // STORE P TO LDS FOR PV MFMA
    // ========================================================================
    s_barrier
    
    // P layout: P[row, col] where row = mfma_row, col determined by MFMA output
    // Each lane owns specific (row, col) positions in the 32×32 P matrix
    // For simplicity, store P in row-major at P_LDS
    
    // MFMA output mapping: lane owns 16 values
    // col = (vreg_idx % 4) + ((lane >> 5) & 1) * 4 + (vreg_idx / 4) * 8
    
    // Store P[mfma_row, col_base + i] for i=0..15
    // col_base depends on lane position
    v_lshrrev_b32_e32 v24, 5, v60         // half = lane >> 5
    v_lshlrev_b32_e32 v24, 2, v24         // half * 4
    
    // P address base: P_LDS + mfma_row * 32 * 4 + col_base * 4
    v_lshlrev_b32_e32 v25, 7, v63         // mfma_row * 128 (32 cols * 4 bytes)
    v_add_u32_e32 v25, P_LDS, v25         // + P_LDS
    
    // Store 16 F32 values
    // v[0:3] → cols 0-3 + half*4
    // v[4:7] → cols 8-11 + half*4
    // etc.
    
    .macro STORE_P_VAL vreg, col_offset
        v_add_u32_e32 v26, \col_offset, v24
        v_lshlrev_b32_e32 v26, 2, v26     // * 4 bytes
        v_add_u32_e32 v26, v25, v26
        ds_write_b32 v26, \vreg
    .endm
    
    STORE_P_VAL v0, 0
    STORE_P_VAL v1, 1
    STORE_P_VAL v2, 2
    STORE_P_VAL v3, 3
    STORE_P_VAL v4, 8
    STORE_P_VAL v5, 9
    STORE_P_VAL v6, 10
    STORE_P_VAL v7, 11
    STORE_P_VAL v8, 16
    STORE_P_VAL v9, 17
    STORE_P_VAL v10, 18
    STORE_P_VAL v11, 19
    STORE_P_VAL v12, 24
    STORE_P_VAL v13, 25
    STORE_P_VAL v14, 26
    STORE_P_VAL v15, 27
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD V TO LDS
    // ========================================================================
    v_add_u32_e32 v54, V_LDS, v51         // V LDS addr
    buffer_load_dwordx4 v[24:27], v50, s[16:19], 0 offen
    s_waitcnt vmcnt(0)
    ds_write_b128 v54, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // PV MFMA: O += P @ V (for one HD tile)
    // ========================================================================
    // P at P_LDS [32×32 F32], V at V_LDS [32×128 FP8]
    // Need to read P row (32 F32), convert to FP8, read V col (32 FP8)
    
    // For K=32, need 2 MFMA passes (32/16 = 2)
    
    // P read base: P_LDS + mfma_row * 32 * 4
    v_lshlrev_b32_e32 v25, 7, v63         // mfma_row * 128
    v_add_u32_e32 v25, P_LDS, v25         // P row base
    
    // V read: need col from V, stride = PITCH
    // V col = determined by MFMA B operand mapping
    // For now, use simple col = lane & 31
    v_and_b32_e32 v26, 31, v60            // V col
    v_lshrrev_b32_e32 v27, 5, v60         // V row block (0 or 1)
    
    // PV pass 0: k=0..15
    // Read P[row, 0:7] as 8 F32, convert to FP8
    ds_read_b128 v[28:31], v25            // P[row, 0:3]
    v_add_u32_e32 v40, 16, v25
    ds_read_b128 v[32:35], v40            // P[row, 4:7]
    s_waitcnt lgkmcnt(0)
    
    // Convert 8 F32 to 8 FP8 packed in 2 dwords
    v_cvt_pk_fp8_f32 v36, v28, v29
    v_and_b32_e32 v36, 0xFFFF, v36
    v_cvt_pk_fp8_f32 v37, v30, v31
    v_lshlrev_b32_e32 v37, 16, v37
    v_or_b32_e32 v36, v36, v37            // v36 = P[0:3] as FP8
    
    v_cvt_pk_fp8_f32 v37, v32, v33
    v_and_b32_e32 v37, 0xFFFF, v37
    v_cvt_pk_fp8_f32 v38, v34, v35
    v_lshlrev_b32_e32 v38, 16, v38
    v_or_b32_e32 v37, v37, v38            // v37 = P[4:7] as FP8
    
    // Read V column (8 FP8 from rows 0-7)
    // V addr = V_LDS + row * PITCH + col
    v_mov_b32_e32 v40, V_LDS
    v_add_u32_e32 v40, v40, v26           // + col
    
    // Read 8 bytes from 8 rows
    .irp row, 0, 1, 2, 3, 4, 5, 6, 7
        v_add_u32_e32 v41, \row * PITCH, v40
        ds_read_u8 v[42+\row], v41
    .endr
    s_waitcnt lgkmcnt(0)
    
    // Pack V bytes
    v_lshlrev_b32_e32 v43, 8, v43
    v_or_b32_e32 v42, v42, v43
    v_lshlrev_b32_e32 v44, 16, v44
    v_or_b32_e32 v42, v42, v44
    v_lshlrev_b32_e32 v45, 24, v45
    v_or_b32_e32 v38, v42, v45            // v38 = V[0:3]
    
    v_lshlrev_b32_e32 v47, 8, v47
    v_or_b32_e32 v46, v46, v47
    v_lshlrev_b32_e32 v48, 16, v48
    v_or_b32_e32 v46, v46, v48
    v_lshlrev_b32_e32 v49, 24, v49
    v_or_b32_e32 v39, v46, v49            // v39 = V[4:7]
    
    // PV MFMA
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], v[36:37], v[38:39], v[80:95]
    s_nop 15
    
    // PV pass 1: k=16..31
    v_add_u32_e32 v40, 32, v25            // P[row, 8:15]
    ds_read_b128 v[28:31], v40
    v_add_u32_e32 v40, 48, v25
    ds_read_b128 v[32:35], v40
    s_waitcnt lgkmcnt(0)
    
    v_cvt_pk_fp8_f32 v36, v28, v29
    v_and_b32_e32 v36, 0xFFFF, v36
    v_cvt_pk_fp8_f32 v37, v30, v31
    v_lshlrev_b32_e32 v37, 16, v37
    v_or_b32_e32 v36, v36, v37
    
    v_cvt_pk_fp8_f32 v37, v32, v33
    v_and_b32_e32 v37, 0xFFFF, v37
    v_cvt_pk_fp8_f32 v38, v34, v35
    v_lshlrev_b32_e32 v38, 16, v38
    v_or_b32_e32 v37, v37, v38
    
    // V rows 8-15
    v_mov_b32_e32 v40, V_LDS
    v_add_u32_e32 v40, v40, v26
    v_add_u32_e32 v40, 8 * PITCH, v40     // Start at row 8
    
    .irp row, 0, 1, 2, 3, 4, 5, 6, 7
        v_add_u32_e32 v41, \row * PITCH, v40
        ds_read_u8 v[42+\row], v41
    .endr
    s_waitcnt lgkmcnt(0)
    
    v_lshlrev_b32_e32 v43, 8, v43
    v_or_b32_e32 v42, v42, v43
    v_lshlrev_b32_e32 v44, 16, v44
    v_or_b32_e32 v42, v42, v44
    v_lshlrev_b32_e32 v45, 24, v45
    v_or_b32_e32 v38, v42, v45
    
    v_lshlrev_b32_e32 v47, 8, v47
    v_or_b32_e32 v46, v46, v47
    v_lshlrev_b32_e32 v48, 16, v48
    v_or_b32_e32 v46, v46, v48
    v_lshlrev_b32_e32 v49, 24, v49
    v_or_b32_e32 v39, v46, v49
    
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], v[36:37], v[38:39], v[80:95]
    s_nop 15
    
    // ========================================================================
    // FINAL NORMALIZATION: O = O / row_sum
    // ========================================================================
    v_rcp_f32_e32 v66, v66
    s_nop 3
    
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v\i, v66
    .endr
    
    // ========================================================================
    // STORE O OUTPUT
    // ========================================================================
    // O[mfma_row, col] where col depends on lane
    // For now, simple store: each lane writes 16 F32 to contiguous location
    
    // v40 = offset from O_ptr base (lane * 64 bytes)
    v_lshlrev_b32_e32 v40, 6, v60         // lane * 64 bytes
    
    // buffer_store with offen: addr = buffer_base + vgpr_offset
    buffer_store_dwordx4 v[80:83], v40, s[4:7], 0 offen
    v_add_u32_e32 v42, 16, v40
    buffer_store_dwordx4 v[84:87], v42, s[4:7], 0 offen
    v_add_u32_e32 v42, 32, v40
    buffer_store_dwordx4 v[88:91], v42, s[4:7], 0 offen
    v_add_u32_e32 v42, 48, v40
    buffer_store_dwordx4 v[92:95], v42, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_kloop_p136E
    .amdhsa_group_segment_fixed_size 20480
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 100
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 100
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter16fwd_fp8_kloop_p136E
    .symbol: _ZN5aiter16fwd_fp8_kloop_p136E.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 20480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 100
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V_ptr, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

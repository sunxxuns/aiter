// TR_B8 + MFMA test
// Store K in column-major (16-byte stride), use TR_B8 to load, run MFMA

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter14exp_tr_b8_mfmaE
.p2align 8
.type _ZN5aiter14exp_tr_b8_mfmaE,@function

_ZN5aiter14exp_tr_b8_mfmaE:
    s_mov_b64 exec, -1
    
    // Args: output, K_ptr, Q_ptr
    s_load_dwordx4 s[4:7], s[0:1], 0
    s_load_dwordx2 s[10:11], s[0:1], 16
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0   // lane_id
    
    // ========================================================================
    // STORE K[32×16] IN COLUMN-MAJOR WITH 16-BYTE ROW STRIDE
    // For TR_B8: K[row, col] at LDS[col + row * 16]
    // Total: 32 rows × 16 cols = 512 bytes
    // ========================================================================
    
    // Only lanes 0-31 load K (each loads one row of 16 bytes)
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    // Global: K_ptr + lane * 16 (row-major input)
    v_lshlrev_b32_e32 v1, 4, v0          // lane * 16
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    // Write to LDS column-major: K[row, col] at LDS[col + row * 16]
    // Each lane writes 16 bytes of row `lane` at columns 0-15
    // For col c: LDS addr = c + lane * 16
    // But we need to transpose: write byte b to LDS[b + lane * 16]
    
    // Actually for column-major with 16-byte row stride:
    // K[row, col] -> LDS[col * 32 + row] (if 32 rows)
    // No wait, TR_B8 reads: output[k] = LDS[(base>>3) + k*16]
    // So we need: LDS[(base>>3) + k*16] = K[row, k]
    // If base = row * 8, then (base>>3) = row
    // So: LDS[row + k*16] = K[row, k]
    
    // Write K[lane, 0:15] to LDS[lane + col*16] for col=0..15
    v_mov_b32_e32 v30, v0                // row = lane
    
    // Extract bytes and write
    // Byte 0-3 from v20
    v_and_b32_e32 v31, 0xff, v20         // K[lane, 0]
    ds_write_b8 v30, v31                 // LDS[lane + 0*16] = LDS[lane]
    
    v_lshrrev_b32_e32 v31, 8, v20
    v_and_b32_e32 v31, 0xff, v31         // K[lane, 1]
    v_add_u32_e32 v32, 16, v30
    ds_write_b8 v32, v31                 // LDS[lane + 1*16]
    
    v_lshrrev_b32_e32 v31, 16, v20
    v_and_b32_e32 v31, 0xff, v31         // K[lane, 2]
    v_add_u32_e32 v32, 32, v30
    ds_write_b8 v32, v31
    
    v_lshrrev_b32_e32 v31, 24, v20       // K[lane, 3]
    v_add_u32_e32 v32, 48, v30
    ds_write_b8 v32, v31
    
    // Byte 4-7 from v21
    v_and_b32_e32 v31, 0xff, v21
    v_add_u32_e32 v32, 64, v30
    ds_write_b8 v32, v31
    
    v_lshrrev_b32_e32 v31, 8, v21
    v_and_b32_e32 v31, 0xff, v31
    v_add_u32_e32 v32, 80, v30
    ds_write_b8 v32, v31
    
    v_lshrrev_b32_e32 v31, 16, v21
    v_and_b32_e32 v31, 0xff, v31
    v_add_u32_e32 v32, 96, v30
    ds_write_b8 v32, v31
    
    v_lshrrev_b32_e32 v31, 24, v21
    v_add_u32_e32 v32, 112, v30
    ds_write_b8 v32, v31
    
    // Byte 8-15 similarly... (for full K=16)
    // For now just test K=8
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD Q[32×8] ROW-MAJOR TO LDS at offset 512
    // (simpler: just use ds_read_b64 for Q)
    // ========================================================================
    
    s_mov_b64 exec, -1
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    v_lshlrev_b32_e32 v1, 3, v0          // lane * 8
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx2 v[20:21], v[10:11]
    s_waitcnt vmcnt(0)
    
    // Q at LDS[512 + lane * 8]
    v_lshlrev_b32_e32 v30, 3, v0
    v_add_u32_e32 v30, 512, v30
    ds_write_b64 v30, v[20:21]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_mov_b64 exec, -1
    
    // ========================================================================
    // MFMA: Use TR_B8 to read K
    // ========================================================================
    
    // MFMA row mapping
    v_and_b32_e32 v1, 3, v0
    v_lshrrev_b32_e32 v2, 3, v0
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v0
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v61, v3, v4            // mfma_row
    
    // TR_B8 base for K: mfma_row * 8 (so base>>3 = mfma_row)
    v_lshlrev_b32_e32 v70, 3, v61
    
    // Q addr: 512 + mfma_row * 8 + k_base
    v_mov_b32_e32 v71, 0
    v_mov_b32_e32 v4, 4                  // k_base for lanes >= 32 (but K=8 so half)
    v_cmp_ge_u32_e64 vcc, v0, 32
    v_cndmask_b32_e32 v71, v71, v4, vcc
    v_lshlrev_b32_e32 v72, 3, v61
    v_add_u32_e32 v72, v72, v71
    v_add_u32_e32 v72, 512, v72
    
    // Clear accumulators
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0
    v_mov_b32_e32 v13, 0
    v_mov_b32_e32 v14, 0
    v_mov_b32_e32 v15, 0
    
    // Read K with TR_B8
    ds_read_b64_tr_b8 v[30:31], v70
    // Read Q with ds_read_b64
    ds_read_b64 v[32:33], v72
    s_waitcnt lgkmcnt(0)
    
    // MFMA (only K=8 for this test)
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
    s_nop 15
    
    // Output
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx4 v[40:41], v[0:3]
    v_add_co_u32_e32 v42, vcc, 16, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[4:7]
    v_add_co_u32_e32 v42, vcc, 32, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[8:11]
    v_add_co_u32_e32 v42, vcc, 48, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[12:15]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter14exp_tr_b8_mfmaE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter14exp_tr_b8_mfmaE
    .symbol: _ZN5aiter14exp_tr_b8_mfmaE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 80
    .max_flat_workgroup_size: 64
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

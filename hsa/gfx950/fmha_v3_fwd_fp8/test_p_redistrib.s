// Test: P redistribution via LDS for correct PV MFMA
//
// Simulates QK MFMA output: thread t has P[Q_rows, K=t%32]
// For simplicity, P[q, k] = (k+1)/16 (normalized, varies by K)
//
// After redistribution:
// Thread t should have P[Q=t%32, K_range] for PV MFMA A operand
//
// With V[k,d] = 1:
// O[q,d] = sum_k(P[q,k] * 1) = sum_k((k+1)/16) = (1+2+...+16)/16 = 136/16 = 8.5

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter15test_p_redistribE
.p2align 8
.type _ZN5aiter15test_p_redistribE,@function

.set LDS_P_OFFSET, 0

_ZN5aiter15test_p_redistribE:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize output
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
    
    // ========================================================================
    // STEP 1: Generate P values as if from QK MFMA
    // Thread t has P values for K column = t % 32
    // P[q, k] = (k+1)/16 for our test (varies by K)
    // We'll store 16 P values per thread (for 16 Q rows)
    // But for 32x32 output, we only use first 32 Q rows
    //
    // Actually, QK MFMA output is interleaved. For simplicity,
    // let's just say thread t owns P values at K = t%32,
    // and we want to redistribute so thread t gets P at Q = t%32
    // ========================================================================
    
    // K = lane_id % 32
    v_and_b32_e32 v1, 31, v0
    // P value = (K+1) / 16 = (K+1) * 0.0625
    v_add_u32_e32 v2, 1, v1              // K+1
    v_cvt_f32_u32_e32 v2, v2             // (K+1) as float
    v_mul_f32_e32 v2, 0x3d800000, v2     // * 0.0625 = /16
    
    // Convert to FP8
    v_mov_b32_e32 v3, 0
    v_cvt_pk_fp8_f32 v3, v2, v2          // 2 identical P values
    
    // ========================================================================
    // STEP 2: Write P to LDS
    // Layout: P[Q_row, K_col] at LDS_P_OFFSET + Q_row * 32 + K_col
    // Each thread writes its P value to its K column for all Q rows
    // Since all Q rows have the same P value per K, we can broadcast
    //
    // For the test, we'll write 32 copies of P at K=lane_id%32
    // ========================================================================
    
    // K_col = lane_id % 32
    v_and_b32_e32 v4, 31, v0
    
    // Write P to all 32 Q rows at K column
    // LDS address = Q_row * 32 + K_col
    // We'll write 1 byte per Q row
    v_add_u32_e32 v5, LDS_P_OFFSET, v4    // Base: LDS_P_OFFSET + K_col
    
    // Write to Q=0..31 (32 bytes total)
    ds_write_b8 v5, v3                    // Q=0
    ds_write_b8 v5, v3 offset:32          // Q=1
    ds_write_b8 v5, v3 offset:64          // Q=2
    ds_write_b8 v5, v3 offset:96          // Q=3
    ds_write_b8 v5, v3 offset:128         // Q=4
    ds_write_b8 v5, v3 offset:160         // Q=5
    ds_write_b8 v5, v3 offset:192         // Q=6
    ds_write_b8 v5, v3 offset:224         // Q=7
    ds_write_b8 v5, v3 offset:256         // Q=8
    ds_write_b8 v5, v3 offset:288         // Q=9
    ds_write_b8 v5, v3 offset:320         // Q=10
    ds_write_b8 v5, v3 offset:352         // Q=11
    ds_write_b8 v5, v3 offset:384         // Q=12
    ds_write_b8 v5, v3 offset:416         // Q=13
    ds_write_b8 v5, v3 offset:448         // Q=14
    ds_write_b8 v5, v3 offset:480         // Q=15
    ds_write_b8 v5, v3 offset:512         // Q=16
    ds_write_b8 v5, v3 offset:544         // Q=17
    ds_write_b8 v5, v3 offset:576         // Q=18
    ds_write_b8 v5, v3 offset:608         // Q=19
    ds_write_b8 v5, v3 offset:640         // Q=20
    ds_write_b8 v5, v3 offset:672         // Q=21
    ds_write_b8 v5, v3 offset:704         // Q=22
    ds_write_b8 v5, v3 offset:736         // Q=23
    ds_write_b8 v5, v3 offset:768         // Q=24
    ds_write_b8 v5, v3 offset:800         // Q=25
    ds_write_b8 v5, v3 offset:832         // Q=26
    ds_write_b8 v5, v3 offset:864         // Q=27
    ds_write_b8 v5, v3 offset:896         // Q=28
    ds_write_b8 v5, v3 offset:928         // Q=29
    ds_write_b8 v5, v3 offset:960         // Q=30
    ds_write_b8 v5, v3 offset:992         // Q=31
    
    s_barrier
    
    // ========================================================================
    // STEP 3: Read P for PV MFMA A operand
    // Thread t needs P[Q=t%32, K_range] where K_range = (t/32)*8:(t/32)*8+8
    // LDS address = Q_row * 32 + K_start
    // ========================================================================
    
    // Q_row = lane_id % 32
    v_and_b32_e32 v6, 31, v0
    // K_start = (lane_id / 32) * 8
    v_lshrrev_b32_e32 v7, 5, v0           // lane_id / 32
    v_lshlrev_b32_e32 v7, 3, v7           // * 8
    
    // LDS address = Q_row * 32 + K_start
    v_lshlrev_b32_e32 v6, 5, v6           // Q_row * 32
    v_add_u32_e32 v6, v6, v7              // + K_start
    v_add_u32_e32 v6, LDS_P_OFFSET, v6    // + LDS base
    
    // Read 8 FP8 values
    ds_read_b64 v[64:65], v6
    
    s_waitcnt lgkmcnt(0)
    
    // Move P to AGPRs for A operand
    v_accvgpr_write_b32 a0, v64
    v_accvgpr_write_b32 a1, v65
    
    // ========================================================================
    // STEP 4: Generate V for B operand
    // V[k,d] = 1 for simplicity
    // ========================================================================
    
    v_mov_b32_e32 v8, 0x3f800000          // 1.0
    v_mov_b32_e32 v68, 0
    v_cvt_pk_fp8_f32 v68, v8, v8
    v_lshlrev_b32_e32 v69, 16, v68
    v_or_b32_e32 v68, v68, v69
    v_mov_b32_e32 v69, v68
    
    s_nop 7
    
    // ========================================================================
    // STEP 5: PV MFMA
    // O[q,d] = sum_k(P[q,k] * V[k,d]) = sum_k(P[q,k] * 1) = sum_k(P[q,k])
    // With P[q,k] = (k+1)/16:
    // O[q,d] = sum_{k=0..15}((k+1)/16) = (1+2+...+16)/16 = 136/16 = 8.5
    // ========================================================================
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[68:69], v[32:47]
    
    s_nop 7
    
    // ========================================================================
    // STEP 6: Store output
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_lshlrev_b32_e32 v3, 6, v0
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_store_dwordx4 v[10:11], v[32:35]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[36:39]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[40:43]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[44:47]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter15test_p_redistribE, .-_ZN5aiter15test_p_redistribE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter15test_p_redistribE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 72
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter15test_p_redistribE
    .symbol: _ZN5aiter15test_p_redistribE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 80
    .agpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - .name: ptr_out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

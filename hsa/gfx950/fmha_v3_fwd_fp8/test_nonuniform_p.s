// Test: Non-uniform P values with P redistribution
// P[q,k] varies by K to test that the correct K values are used in the reduction
//
// Setup:
// P[q, k] = (k+1)/136 (normalized, varies by K)
// sum_k(P[q,k]) = sum((k+1)/136 for k=0..15) = 136/136 = 1.0
// V[k, d] = 1.0 for all positions
//
// Expected: O[q, d] = sum_k(P[q,k] * 1) = 1.0 for all (q,d)
//
// If P redistribution is wrong, different Q rows would see different sums.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter15test_nonuniform_pE
.p2align 8
.type _ZN5aiter15test_nonuniform_pE,@function

.set LDS_P_OFFSET, 0

_ZN5aiter15test_nonuniform_pE:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize accumulators
    v_mov_b32_e32 v48, 0
    v_mov_b32_e32 v49, 0
    v_mov_b32_e32 v50, 0
    v_mov_b32_e32 v51, 0
    v_mov_b32_e32 v52, 0
    v_mov_b32_e32 v53, 0
    v_mov_b32_e32 v54, 0
    v_mov_b32_e32 v55, 0
    v_mov_b32_e32 v56, 0
    v_mov_b32_e32 v57, 0
    v_mov_b32_e32 v58, 0
    v_mov_b32_e32 v59, 0
    v_mov_b32_e32 v60, 0
    v_mov_b32_e32 v61, 0
    v_mov_b32_e32 v62, 0
    v_mov_b32_e32 v63, 0
    
    // ========================================================================
    // Generate P values: P[q,k] = (k+1)/136
    // Thread t writes to K column = t % 32 (but only K=0..15 are used)
    // For K >= 16, we write 0 (not used in the 16-element reduction)
    // ========================================================================
    
    v_and_b32_e32 v1, 31, v0              // K = tid % 32
    
    // If K < 16: P = (K+1)/136, else P = 0
    v_cmp_lt_u32_e64 s[0:1], v1, 16
    v_add_u32_e32 v2, 1, v1               // K+1
    v_cvt_f32_u32_e32 v2, v2
    v_mul_f32_e32 v2, 0x3bf0f0f1, v2      // * (1/136 = 0.00735)
    v_cndmask_b32_e64 v2, 0, v2, s[0:1]   // P = (K<16) ? (K+1)/136 : 0
    
    // Convert to FP8 (OCP e4m3fn format)
    v_mov_b32_e32 v3, 0
    v_cvt_pk_fp8_f32 v3, v2, v2
    
    // Write to LDS: P[Q, K] at Q*32 + K
    v_add_u32_e32 v5, LDS_P_OFFSET, v1
    
    // Write P to all 32 Q rows
    ds_write_b8 v5, v3
    ds_write_b8 v5, v3 offset:32
    ds_write_b8 v5, v3 offset:64
    ds_write_b8 v5, v3 offset:96
    ds_write_b8 v5, v3 offset:128
    ds_write_b8 v5, v3 offset:160
    ds_write_b8 v5, v3 offset:192
    ds_write_b8 v5, v3 offset:224
    ds_write_b8 v5, v3 offset:256
    ds_write_b8 v5, v3 offset:288
    ds_write_b8 v5, v3 offset:320
    ds_write_b8 v5, v3 offset:352
    ds_write_b8 v5, v3 offset:384
    ds_write_b8 v5, v3 offset:416
    ds_write_b8 v5, v3 offset:448
    ds_write_b8 v5, v3 offset:480
    ds_write_b8 v5, v3 offset:512
    ds_write_b8 v5, v3 offset:544
    ds_write_b8 v5, v3 offset:576
    ds_write_b8 v5, v3 offset:608
    ds_write_b8 v5, v3 offset:640
    ds_write_b8 v5, v3 offset:672
    ds_write_b8 v5, v3 offset:704
    ds_write_b8 v5, v3 offset:736
    ds_write_b8 v5, v3 offset:768
    ds_write_b8 v5, v3 offset:800
    ds_write_b8 v5, v3 offset:832
    ds_write_b8 v5, v3 offset:864
    ds_write_b8 v5, v3 offset:896
    ds_write_b8 v5, v3 offset:928
    ds_write_b8 v5, v3 offset:960
    ds_write_b8 v5, v3 offset:992
    
    s_barrier
    
    // ========================================================================
    // Read P for PV MFMA A operand: P[Q=tid%32, K_range]
    // ========================================================================
    
    v_and_b32_e32 v6, 31, v0
    v_lshrrev_b32_e32 v7, 5, v0
    v_lshlrev_b32_e32 v7, 3, v7
    v_lshlrev_b32_e32 v6, 5, v6
    v_add_u32_e32 v6, v6, v7
    v_add_u32_e32 v6, LDS_P_OFFSET, v6
    
    ds_read_b64 v[32:33], v6
    
    s_waitcnt lgkmcnt(0)
    
    v_accvgpr_write_b32 a0, v32
    v_accvgpr_write_b32 a1, v33
    
    // ========================================================================
    // V = 1.0 for all positions
    // ========================================================================
    
    v_mov_b32_e32 v8, 0x3f800000          // 1.0
    v_mov_b32_e32 v64, 0
    v_cvt_pk_fp8_f32 v64, v8, v8
    v_lshlrev_b32_e32 v65, 16, v64
    v_or_b32_e32 v64, v64, v65
    v_mov_b32_e32 v65, v64
    
    s_nop 7
    
    // ========================================================================
    // PV MFMA
    // O[q,d] = sum_k(P[q,k] * V[k,d]) = sum_k(P[q,k] * 1) = sum_k(P[q,k])
    // With P[q,k] = (k+1)/136 for k=0..15:
    // O[q,d] = sum((k+1)/136 for k=0..15) = (1+2+...+16)/136 = 136/136 = 1.0
    // ========================================================================
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    s_nop 7
    
    // ========================================================================
    // Scatter store based on MFMA output positions
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4
    v_lshrrev_b32_e32 v4, 5, v0
    v_lshlrev_b32_e32 v4, 2, v4           // M_base * 4
    
    // Store v48-v51 to M_base + 0,1,2,3
    v_lshlrev_b32_e32 v5, 7, v4
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v48
    flat_store_dword v[12:13], v49 offset:128
    flat_store_dword v[12:13], v50 offset:256
    flat_store_dword v[12:13], v51 offset:384
    
    // Store v52-v55 to M_base + 8,9,10,11
    v_add_u32_e32 v6, 8, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v52
    flat_store_dword v[12:13], v53 offset:128
    flat_store_dword v[12:13], v54 offset:256
    flat_store_dword v[12:13], v55 offset:384
    
    // Store v56-v59 to M_base + 16,17,18,19
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v56
    flat_store_dword v[12:13], v57 offset:128
    flat_store_dword v[12:13], v58 offset:256
    flat_store_dword v[12:13], v59 offset:384
    
    // Store v60-v63 to M_base + 24,25,26,27
    v_add_u32_e32 v6, 24, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v60
    flat_store_dword v[12:13], v61 offset:128
    flat_store_dword v[12:13], v62 offset:256
    flat_store_dword v[12:13], v63 offset:384
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter15test_nonuniform_pE, .-_ZN5aiter15test_nonuniform_pE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter15test_nonuniform_pE
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 68
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter15test_nonuniform_pE
    .symbol: _ZN5aiter15test_nonuniform_pE.kd
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

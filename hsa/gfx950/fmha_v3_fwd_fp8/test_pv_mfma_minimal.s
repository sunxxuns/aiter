// Minimal test: just V×P MFMA with known input patterns
// Input: V=1, P=1 -> Output should be K (16 in this case)
// Or: V[D, K] = D, P[Q, K] = 1 -> Output[Q, D] should be D*K = D*16

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter18test_pv_mfma_minE
.p2align 8
.type _ZN5aiter18test_pv_mfma_minE,@function

_ZN5aiter18test_pv_mfma_minE:
    // Args: ptr_out (8 bytes), ptr_V (8 bytes), ptr_P (8 bytes), mode (4 bytes)
    // Mode 0: V=1, P=1 -> O should be 16 (K dimension)
    // Mode 1: V=D, P=1 -> O[Q, D] should be D*16
    // Mode 2: V=1, P=K/32 -> O should be sum(K/32 for K=0..15) = (0+1+...+15)/32 = 120/32 = 3.75
    
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_out
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_V (ignored, we generate V)
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // ptr_P (ignored, we generate P)
    s_load_dword s14, s[0:1], 0x18         // mode
    
    v_and_b32_e32 v0, 63, v0               // lane_id (0-63)
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize output accumulators
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
    // Generate V values for A operand
    // For mode 0: V = 1.0
    // For mode 1: V = D/32 (where D = lane_id % 32)
    // For mode 2: V = 1.0
    // A operand: Thread t provides V^T[D=t%32, K_range]
    // Each thread needs 8 K values at its D position
    // ========================================================================
    
    v_and_b32_e32 v1, 31, v0              // D = lane_id % 32
    v_cvt_f32_u32_e32 v2, v1              // D as float
    v_mul_f32_e32 v2, 0x3d000000, v2      // D * (1/32) = D/32
    
    // For mode 0 or 2: use 1.0
    // For mode 1: use D/32
    s_cmp_eq_u32 s14, 1
    s_cselect_b32 s15, 0x3f800000, 0x3f800000  // Both are 1.0 for now
    v_mov_b32_e32 v3, s15                 // Default V value = 1.0
    s_cmp_eq_u32 s14, 1
    s_cbranch_scc0 USE_V_ONE
    v_mov_b32_e32 v3, v2                  // V = D/32 for mode 1
USE_V_ONE:
    
    // Convert V float to FP8 and replicate to 8 values
    // v_cvt_pk_fp8_f32 produces 2 FP8 values in lower 16 bits
    v_mov_b32_e32 v4, 0
    v_cvt_pk_fp8_f32 v4, v3, v3           // 2 identical V values
    v_lshlrev_b32_e32 v5, 16, v4
    v_or_b32_e32 v4, v4, v5               // 4 identical V values
    // Now v4 has 4 identical FP8 V values
    v_mov_b32_e32 v5, v4                  // v5 also has 4 V values
    // a[0:1] = 8 V values
    v_accvgpr_write_b32 a0, v4
    v_accvgpr_write_b32 a1, v5
    
    // ========================================================================
    // Generate P values for B operand
    // For mode 0 or 1: P = 1.0
    // For mode 2: P = K/32 where K is the K position (0..15 for this tile)
    // B operand: Thread t provides P^T[K_range, Q=t%32]
    // Each thread needs 8 K values at its Q position
    // For mode 0/1: all P = 1.0
    // ========================================================================
    
    // For simplicity, use P = 1.0 for all modes initially
    v_mov_b32_e32 v6, 0x3f800000          // P = 1.0
    v_mov_b32_e32 v7, 0
    v_cvt_pk_fp8_f32 v7, v6, v6           // 2 P values
    v_lshlrev_b32_e32 v8, 16, v7
    v_or_b32_e32 v7, v7, v8               // 4 P values
    v_mov_b32_e32 v8, v7                  // Another 4 P values
    // v[64:65] = 8 P values
    v_mov_b32_e32 v64, v7
    v_mov_b32_e32 v65, v8
    
    s_nop 7
    
    // ========================================================================
    // V×P MFMA: A=V, B=P
    // Computes O^T[D, Q] = V^T[D, K] × P^T[K, Q]
    // With V=1 and P=1: O^T should be K = 16 (for 16 K values reduced)
    // ========================================================================
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 7
    
    // ========================================================================
    // Store output
    // Thread t owns O^T[D_base:D_base+16, Q=t%32] where D_base = (t/32)*16
    // Store as O[Q, D] at output_ptr + (t%32)*128*4 + (t/32)*16*4
    //        = output_ptr + (t%32)*512 + (t/32)*64
    // ========================================================================
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0              // Q = t%32
    v_lshrrev_b32_e32 v4, 5, v0           // D_group = t/32
    v_lshlrev_b32_e32 v3, 9, v3           // Q * 512
    v_lshlrev_b32_e32 v4, 6, v4           // D_group * 64
    v_add_u32_e32 v3, v3, v4
    
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

.size _ZN5aiter18test_pv_mfma_minE, .-_ZN5aiter18test_pv_mfma_minE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter18test_pv_mfma_minE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 20
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
  - .name: _ZN5aiter18test_pv_mfma_minE
    .symbol: _ZN5aiter18test_pv_mfma_minE.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 80
    .agpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - .name: ptr_out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_V
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_P
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
      - .name: mode
        .size: 4
        .offset: 24
        .value_kind: by_value
...
.end_amdgpu_metadata

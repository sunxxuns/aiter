// FP8 Flash Attention with Pitch-136 LDS layout
// 256 threads (4 waves), bank-conflict-free access
// Computes O = softmax(Q @ K^T / sqrt(d)) @ V

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set PITCH, 136                     // Row pitch for zero bank conflicts
.set Q_LDS, 0                       // Q at LDS[0]
.set K_LDS, 4352                    // K after Q (32 * 136)
.set P_LDS, 8704                    // P after K
.set V_LDS, 12800                   // V after P (need space for P 32*32*4=4096)
.set LDS_SIZE, 17152                // Total LDS needed
.set SCALE, 0x3e028f5c              // log2(e) / sqrt(128)

.text
.globl _ZN5aiter11fwd_fp8_p136E
.p2align 8
.type _ZN5aiter11fwd_fp8_p136E,@function

_ZN5aiter11fwd_fp8_p136E:
    s_mov_b64 exec, -1
    
    // Args: O_ptr, Q_ptr, K_ptr, V_ptr, seq_len
    s_load_dwordx4 s[4:7], s[0:1], 0      // O, Q
    s_load_dwordx4 s[12:15], s[0:1], 16   // K, V
    s_load_dword s24, s[0:1], 32          // seq_len
    s_waitcnt lgkmcnt(0)
    
    v_and_b32_e32 v60, 63, v0             // lane_id
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id
    v_and_b32_e32 v62, 255, v0            // tid (0-255)
    
    s_mov_b32 s2, SCALE
    
    // ========================================================================
    // LOAD Q[32×128] TO LDS WITH PITCH-136
    // Each thread loads 16 bytes (128 bytes/row × 32 rows = 4096 bytes total)
    // 256 threads load 256 × 16 = 4096 bytes
    // Row = tid / 8, col_chunk = tid % 8
    // Global addr: row * 128 + col_chunk * 16
    // LDS addr: row * 136 + col_chunk * 16
    // ========================================================================
    
    v_lshrrev_b32_e32 v1, 3, v62          // row = tid / 8
    v_and_b32_e32 v2, 7, v62              // col_chunk = tid % 8
    
    // Global Q offset: row * 128 + col_chunk * 16
    v_lshlrev_b32_e32 v3, 7, v1           // row * 128
    v_lshlrev_b32_e32 v4, 4, v2           // col_chunk * 16
    v_add_u32_e32 v3, v3, v4              // global offset
    
    // LDS Q offset: row * 136 + col_chunk * 16
    v_mov_b32_e32 v5, PITCH
    v_mul_lo_u32 v6, v1, v5               // row * 136
    v_add_u32_e32 v6, v6, v4              // + col_chunk * 16
    v_add_u32_e32 v6, Q_LDS, v6           // + Q base
    
    // Global Q address
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K[32×128] TO LDS WITH PITCH-136
    // Same pattern as Q
    // ========================================================================
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v3, v10    // Reuse same global offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS K offset
    v_mul_lo_u32 v7, v1, v5               // row * 136
    v_add_u32_e32 v7, v7, v4
    v_add_u32_e32 v7, K_LDS, v7           // + K base
    
    flat_load_dwordx4 v[24:27], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v7, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE MFMA ROW MAPPING
    // ========================================================================
    
    v_and_b32_e32 v1, 3, v60              // lane & 3
    v_lshrrev_b32_e32 v2, 3, v60          // lane >> 3
    v_and_b32_e32 v2, 3, v2               // & 3
    v_lshlrev_b32_e32 v2, 2, v2           // * 4
    v_add_u32_e32 v3, v1, v2              // row16
    
    v_lshrrev_b32_e32 v4, 2, v60          // lane >> 2
    v_and_b32_e32 v4, 1, v4               // & 1
    v_lshlrev_b32_e32 v4, 4, v4           // * 16
    v_add_u32_e32 v63, v3, v4             // mfma_row
    
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc      // k_half
    
    // Q read base: Q_LDS + mfma_row * PITCH + k_half
    v_mov_b32_e32 v1, PITCH
    v_mul_lo_u32 v70, v63, v1
    v_add_u32_e32 v70, v70, v64
    // v70 = mfma_row * PITCH + k_half (offset from base)
    
    // K read base: K_LDS + mfma_row * PITCH + k_half
    v_add_u32_e32 v71, K_LDS, v70         // v71 = K read addr
    v_add_u32_e32 v70, Q_LDS, v70         // v70 = Q read addr (add Q base)
    
    // ========================================================================
    // CLEAR ACCUMULATORS AND RUN QK MFMA
    // ========================================================================
    
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // 8 iterations for HD=128 (128/16 = 8)
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
    
    // S is now in v[0:15]
    
    // ========================================================================
    // SOFTMAX
    // ========================================================================
    
    // Row max
    v_max_f32_e32 v20, v0, v1
    v_max_f32_e32 v20, v20, v2
    v_max_f32_e32 v20, v20, v3
    v_max_f32_e32 v20, v20, v4
    v_max_f32_e32 v20, v20, v5
    v_max_f32_e32 v20, v20, v6
    v_max_f32_e32 v20, v20, v7
    v_max_f32_e32 v20, v20, v8
    v_max_f32_e32 v20, v20, v9
    v_max_f32_e32 v20, v20, v10
    v_max_f32_e32 v20, v20, v11
    v_max_f32_e32 v20, v20, v12
    v_max_f32_e32 v20, v20, v13
    v_max_f32_e32 v20, v20, v14
    v_max_f32_e32 v20, v20, v15
    
    v_mov_b32_e32 v21, v20
    s_nop 1
    v_permlane32_swap_b32_e32 v21, v20
    v_max_f32_e32 v20, v20, v21
    
    // exp2((S - max) * scale)
    v_mul_f32_e32 v21, s2, v20
    
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_fma_f32 v\i, v\i, s2, -v21
        v_exp_f32_e32 v\i, v\i
    .endr
    s_nop 7
    s_nop 7
    s_nop 7
    
    // Row sum
    v_add_f32_e32 v22, v0, v1
    .irp i, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_add_f32_e32 v22, v22, v\i
    .endr
    
    v_mov_b32_e32 v23, v22
    s_nop 1
    v_permlane32_swap_b32_e32 v23, v22
    v_add_f32_e32 v22, v22, v23
    
    // Normalize
    v_rcp_f32_e32 v22, v22
    s_nop 3
    
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mul_f32_e32 v\i, v\i, v22
    .endr
    
    // P is now in v[0:15]
    
    // ========================================================================
    // STORE OUTPUT (just P for now to verify softmax works)
    // ========================================================================
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60         // lane * 64
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
.amdhsa_kernel _ZN5aiter11fwd_fp8_p136E
    .amdhsa_group_segment_fixed_size 20480
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 28
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter11fwd_fp8_p136E
    .symbol: _ZN5aiter11fwd_fp8_p136E.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 20480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 28
    .vgpr_count: 80
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V_ptr, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

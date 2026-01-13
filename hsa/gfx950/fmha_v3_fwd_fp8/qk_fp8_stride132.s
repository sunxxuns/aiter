// FP8 QK MFMA with stride-132 LDS layout for ZERO bank conflicts
// Computes S^T = K @ Q^T for HD=128 (8 MFMAs)
//
// Key insight: Using LDS row stride 132 (not 128) spreads bank accesses
// - Stride 128: all rows hit same bank (128 = 32 banks × 4 bytes)
// - Stride 132: rows hit different banks (132 % 128 = 4 bytes offset)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

// Constants
.set LDS_STRIDE, 132         // Row stride for bank-conflict-free access
.set HD, 128                 // Head dimension
.set Q_ROWS, 32              // Q matrix rows
.set K_ROWS, 32              // K matrix rows per tile
.set Q_LDS_BASE, 0           // Q starts at LDS[0]
.set K_LDS_BASE, 4224        // K starts after Q (32*132 = 4224, aligned to 8)

.text
.globl _ZN5aiter15qk_fp8_stride132E
.p2align 8
.type _ZN5aiter15qk_fp8_stride132E,@function

_ZN5aiter15qk_fp8_stride132E:
    s_mov_b64 exec, -1
    
    // Args: output, K_ptr, Q_ptr
    s_load_dwordx4 s[4:7], s[0:1], 0     // output, K_ptr
    s_load_dwordx2 s[10:11], s[0:1], 16  // Q_ptr
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0   // lane_id
    
    // ========================================================================
    // LOAD Q[32×128] TO LDS WITH STRIDE 132
    // Each of 32 lanes loads one row (128 bytes) using multiple loads
    // ========================================================================
    
    // Only lanes 0-31 load Q
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    // Global address: Q_ptr + lane * 128
    v_lshlrev_b32_e32 v1, 7, v0         // lane * 128
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS address: Q_LDS_BASE + lane * LDS_STRIDE
    v_mov_b32_e32 v2, LDS_STRIDE
    v_mul_lo_u32 v5, v0, v2             // lane * 132
    v_add_u32_e32 v5, Q_LDS_BASE, v5    // + Q base
    
    // Load 128 bytes in 8 × 16-byte loads, write with stride 132
    .irp off, 0, 16, 32, 48, 64, 80, 96, 112
        flat_load_dwordx4 v[20:23], v[10:11] offset:\off
        s_waitcnt vmcnt(0)
        v_add_u32_e32 v6, \off, v5
        ds_write_b128 v6, v[20:23]
    .endr
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K[32×128] TO LDS WITH STRIDE 132
    // ========================================================================
    
    s_mov_b64 exec, -1
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    // Global address: K_ptr + lane * 128
    v_lshlrev_b32_e32 v1, 7, v0
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS address: K_LDS_BASE + lane * LDS_STRIDE
    v_mov_b32_e32 v2, LDS_STRIDE
    v_mul_lo_u32 v5, v0, v2
    v_add_u32_e32 v5, K_LDS_BASE, v5
    
    // Load 128 bytes
    .irp off, 0, 16, 32, 48, 64, 80, 96, 112
        flat_load_dwordx4 v[20:23], v[10:11] offset:\off
        s_waitcnt vmcnt(0)
        v_add_u32_e32 v6, \off, v5
        ds_write_b128 v6, v[20:23]
    .endr
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_mov_b64 exec, -1
    
    // ========================================================================
    // COMPUTE MFMA ROW MAPPING
    // row = (lane & 3) + ((lane >> 3) & 3) * 4 + ((lane >> 2) & 1) * 16
    // ========================================================================
    
    v_and_b32_e32 v1, 3, v0              // lane & 3
    v_lshrrev_b32_e32 v2, 3, v0          // lane >> 3
    v_and_b32_e32 v2, 3, v2              // & 3
    v_lshlrev_b32_e32 v2, 2, v2          // * 4
    v_add_u32_e32 v3, v1, v2             // row16
    
    v_lshrrev_b32_e32 v4, 2, v0          // lane >> 2
    v_and_b32_e32 v4, 1, v4              // & 1 = row_hi
    v_lshlrev_b32_e32 v4, 4, v4          // row_hi * 16
    v_add_u32_e32 v61, v3, v4            // v61 = full_row (0-31)
    
    // k_base: 0 for lanes 0-31, 8 for lanes 32-63
    v_mov_b32_e32 v62, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v0, 32
    v_cndmask_b32_e32 v62, v62, v4, vcc  // v62 = k_base
    
    // Compute base LDS addresses for K and Q
    // K_addr = K_LDS_BASE + row * LDS_STRIDE + k_base
    // Q_addr = Q_LDS_BASE + row * LDS_STRIDE + k_base (same row mapping for Q^T)
    v_mov_b32_e32 v2, LDS_STRIDE
    v_mul_lo_u32 v63, v61, v2            // row * 132
    v_add_u32_e32 v63, v63, v62          // + k_base
    
    // v70 = K LDS base address for this lane
    v_add_u32_e32 v70, K_LDS_BASE, v63
    // v71 = Q LDS base address for this lane
    v_add_u32_e32 v71, Q_LDS_BASE, v63
    
    // ========================================================================
    // CLEAR ACCUMULATORS (v0-v15)
    // ========================================================================
    
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
    
    // ========================================================================
    // 8 MFMA ITERATIONS FOR HD=128
    // Each iteration: K=16, total K=128
    // ========================================================================
    
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        // Read K operand (A): 8 bytes at K_addr + k_off
        v_add_u32_e32 v72, \k_off, v70
        ds_read_b64 v[30:31], v72
        
        // Read Q operand (B): 8 bytes at Q_addr + k_off
        v_add_u32_e32 v73, \k_off, v71
        ds_read_b64 v[32:33], v73
        
        s_waitcnt lgkmcnt(0)
        
        // MFMA: accumulator += K * Q^T
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    
    s_nop 15
    
    // ========================================================================
    // OUTPUT RESULTS
    // ========================================================================
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60        // lane * 64 (16 floats × 4 bytes)
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
.amdhsa_kernel _ZN5aiter15qk_fp8_stride132E
    .amdhsa_group_segment_fixed_size 16384
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
  - .name: _ZN5aiter15qk_fp8_stride132E
    .symbol: _ZN5aiter15qk_fp8_stride132E.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 16384
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

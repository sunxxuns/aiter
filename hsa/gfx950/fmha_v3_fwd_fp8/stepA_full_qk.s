// Step A: Full QK kernel with ds_read_b64 + XOR swizzle baseline
// Computes S^T = K @ Q^T where K is 32×128 (using first 16 cols), Q is 32×128

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

// XOR swizzle macro
.macro XOR_SWIZZLE dst, src, tmp
    v_and_b32_e32 \tmp, 0x1ff, \src
    v_lshrrev_b32_e32 \tmp, 7, \tmp
    v_lshlrev_b32_e32 \tmp, 3, \tmp
    v_xor_b32_e32 \dst, \tmp, \src
.endm

.text
.globl _ZN5aiter12stepA_full_qkE
.p2align 8
.type _ZN5aiter12stepA_full_qkE,@function

_ZN5aiter12stepA_full_qkE:
    s_mov_b64 exec, -1
    
    // Args: output, K_ptr, Q_ptr
    s_load_dwordx4 s[4:7], s[0:1], 0     // output, K_ptr
    s_load_dwordx2 s[10:11], s[0:1], 16  // Q_ptr
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0   // lane_id
    
    // ========================================================================
    // Load K matrix (32×16) to LDS[0:511] - row major with swizzle
    // First 32 lanes load, each loads 16 bytes (one row)
    // ========================================================================
    
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    // Global load K[lane, 0:15]
    v_lshlrev_b32_e32 v1, 7, v0         // lane * 128 (row stride in global)
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]  // Load 16 bytes
    s_waitcnt vmcnt(0)
    
    // Write to LDS with swizzle: K at LDS[0:511]
    v_lshlrev_b32_e32 v5, 4, v0         // lane * 16
    XOR_SWIZZLE v6, v5, v7
    ds_write_b64 v6, v[20:21]
    v_add_u32_e32 v8, 8, v5
    XOR_SWIZZLE v6, v8, v7
    ds_write_b64 v6, v[22:23]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // Load Q matrix (32×16) to LDS[512:1023] - row major with swizzle
    // ========================================================================
    
    s_mov_b64 exec, -1
    v_cmp_gt_u32_e64 vcc, 32, v0
    s_and_saveexec_b64 s[12:13], vcc
    
    // Global load Q[lane, 0:15]
    v_lshlrev_b32_e32 v1, 7, v0         // lane * 128
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[24:27], v[10:11]
    s_waitcnt vmcnt(0)
    
    // Write to LDS[512:1023] with swizzle
    v_lshlrev_b32_e32 v5, 4, v0
    v_add_u32_e32 v5, 512, v5           // Q base at 512
    XOR_SWIZZLE v6, v5, v7
    ds_write_b64 v6, v[24:25]
    v_add_u32_e32 v8, 8, v5
    XOR_SWIZZLE v6, v8, v7
    ds_write_b64 v6, v[26:27]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    s_mov_b64 exec, -1
    
    // ========================================================================
    // Compute MFMA operand addresses using correct row mapping
    // A operand (K): row = row16 + row_hi * 16
    // B operand (Q^T): similar mapping for columns
    // ========================================================================
    
    // A operand (K) address
    v_and_b32_e32 v1, 3, v0              // lane & 3
    v_lshrrev_b32_e32 v2, 3, v0          // lane >> 3
    v_and_b32_e32 v2, 3, v2              // & 3
    v_lshlrev_b32_e32 v2, 2, v2          // * 4
    v_add_u32_e32 v3, v1, v2             // row16
    
    v_lshrrev_b32_e32 v4, 2, v0          // lane >> 2
    v_and_b32_e32 v4, 1, v4              // & 1 = row_hi
    v_lshlrev_b32_e32 v4, 4, v4          // row_hi * 16
    v_add_u32_e32 v3, v3, v4             // full_row = row16 + row_hi*16
    
    // k_base for A
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 8
    v_cmp_ge_u32_e64 vcc, v0, 32
    v_cndmask_b32_e32 v4, v4, v5, vcc    // k_base = 0 or 8
    
    // A LDS address = row * 16 + k_base
    v_lshlrev_b32_e32 v70, 4, v3
    v_add_u32_e32 v70, v70, v4
    XOR_SWIZZLE v70, v70, v7
    
    // B operand (Q^T) - for now use same mapping (B mirrors A for symmetric case)
    // For Q^T, we need col mapping. Using same formula for simplicity in test.
    v_add_u32_e32 v71, 512, v70          // Q base at 512
    
    // Read A (K) operand
    ds_read_b64 v[30:31], v70
    // Read B (Q^T) operand  
    ds_read_b64 v[32:33], v71
    s_waitcnt lgkmcnt(0)
    
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
    
    // MFMA: C = A * B = K * Q^T
    v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
    s_nop 15
    s_nop 15
    
    // Output
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60        // lane * 64 (16 floats)
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
.amdhsa_kernel _ZN5aiter12stepA_full_qkE
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
  - .name: _ZN5aiter12stepA_full_qkE
    .symbol: _ZN5aiter12stepA_full_qkE.kd
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

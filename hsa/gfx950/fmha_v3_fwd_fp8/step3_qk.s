// Step 3: QK MFMA using simple row-major LDS
// Compute S^T = K @ Q^T using FP8 MFMA (transposed to avoid K transpose)
// Q: 32×128, K: 32×128, S^T: 32×32
// Output is S^T (transposed attention scores)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter8step3_qkE
.p2align 8
.type _ZN5aiter8step3_qkE,@function

_ZN5aiter8step3_qkE:
    s_mov_b64 exec, -1
    
    // Args: [output_ptr, Q_ptr, K_ptr]
    s_load_dwordx4 s[4:7], s[0:1], 0      // s[4:5]=output, s[6:7]=Q
    s_load_dwordx2 s[8:9], s[0:1], 16     // s[8:9]=K
    s_waitcnt lgkmcnt(0)
    
    // v0 = thread ID (0-63 for wave 0)
    v_mov_b32_e32 v60, v0
    
    // ========================================================================
    // Load Q to LDS[0:4095] (32×128 = 4096 bytes)
    // Each of 64 threads loads 64 bytes
    // ========================================================================
    
    v_lshrrev_b32_e32 v1, 1, v0           // row = tid / 2
    v_and_b32_e32 v2, 1, v0               // col_half = tid & 1
    v_lshlrev_b32_e32 v3, 6, v2           // col_offset = col_half * 64
    v_lshlrev_b32_e32 v4, 7, v1           // row_offset = row * 128
    v_add_u32_e32 v4, v3, v4              // global offset
    
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_load_dwordx4 v[24:27], v[12:13]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_load_dwordx4 v[28:31], v[12:13]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_load_dwordx4 v[32:35], v[12:13]
    s_waitcnt vmcnt(0)
    
    // Write Q to LDS[0:4095]
    v_lshlrev_b32_e32 v5, 6, v0           // LDS offset = tid * 64
    ds_write_b128 v5, v[20:23]
    v_add_u32_e32 v6, 16, v5
    ds_write_b128 v6, v[24:27]
    v_add_u32_e32 v6, 32, v5
    ds_write_b128 v6, v[28:31]
    v_add_u32_e32 v6, 48, v5
    ds_write_b128 v6, v[32:35]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // Load K to LDS[4096:8191] (32×128 = 4096 bytes)
    // Same pattern as Q
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[36:39], v[10:11]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_load_dwordx4 v[40:43], v[12:13]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_load_dwordx4 v[44:47], v[12:13]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_load_dwordx4 v[48:51], v[12:13]
    s_waitcnt vmcnt(0)
    
    // Write K to LDS[4096:8191]
    v_add_u32_e32 v5, 4096, v5            // LDS offset += 4096
    ds_write_b128 v5, v[36:39]
    v_add_u32_e32 v6, 16, v5
    ds_write_b128 v6, v[40:43]
    v_add_u32_e32 v6, 32, v5
    ds_write_b128 v6, v[44:47]
    v_add_u32_e32 v6, 48, v5
    ds_write_b128 v6, v[48:51]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // Compute S^T = K @ Q^T using FP8 MFMA
    // MFMA 32×32×16: 8 iterations for HD=128
    // A operand = K (in AGPRs), B operand = Q
    // ========================================================================
    
    // Initialize accumulators to 0
    v_mov_b32_e32 v96, 0
    v_mov_b32_e32 v97, 0
    v_mov_b32_e32 v98, 0
    v_mov_b32_e32 v99, 0
    v_mov_b32_e32 v100, 0
    v_mov_b32_e32 v101, 0
    v_mov_b32_e32 v102, 0
    v_mov_b32_e32 v103, 0
    v_mov_b32_e32 v104, 0
    v_mov_b32_e32 v105, 0
    v_mov_b32_e32 v106, 0
    v_mov_b32_e32 v107, 0
    v_mov_b32_e32 v108, 0
    v_mov_b32_e32 v109, 0
    v_mov_b32_e32 v110, 0
    v_mov_b32_e32 v111, 0
    
    // LDS read addresses (same as fwd_fp8_kloop.s)
    // Each thread reads from row (tid & 31), with half (tid >> 5) selecting byte offset
    v_and_b32_e32 v2, 31, v0              // row = tid & 31
    v_lshrrev_b32_e32 v3, 5, v0           // half = tid >> 5
    v_lshlrev_b32_e32 v70, 7, v2          // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v70, v70, v4            // Q base = row*128 + half*8
    v_add_u32_e32 v71, 4096, v70          // K base = Q base + 4096
    
    // 8 MFMA iterations for HD=128 (16 bytes per iteration)
    .irp hd_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \hd_off, v71    // K offset (A operand)
        v_add_u32_e32 v73, \hd_off, v70    // Q offset (B operand)
        ds_read_b64 v[30:31], v72          // Read K for A operand
        ds_read_b64 v[34:35], v73          // Read Q for B operand
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v30        // K → AGPR
        v_accvgpr_write_b32 a1, v31
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[96:111], a[0:1], v[34:35], v[96:111]
        s_nop 15
    .endr
    
    // ========================================================================
    // Output S^T (64 threads × 16 floats = 1024 floats)
    // ========================================================================
    
    v_lshlrev_b32_e32 v70, 6, v60         // output offset = tid * 64
    v_mov_b32_e32 v72, s4
    v_mov_b32_e32 v73, s5
    v_add_co_u32_e32 v72, vcc, v70, v72
    v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
    
    flat_store_dwordx4 v[72:73], v[96:99]
    v_add_co_u32_e32 v74, vcc, 16, v72
    v_addc_co_u32_e32 v75, vcc, 0, v73, vcc
    flat_store_dwordx4 v[74:75], v[100:103]
    v_add_co_u32_e32 v74, vcc, 32, v72
    v_addc_co_u32_e32 v75, vcc, 0, v73, vcc
    flat_store_dwordx4 v[74:75], v[104:107]
    v_add_co_u32_e32 v74, vcc, 48, v72
    v_addc_co_u32_e32 v75, vcc, 0, v73, vcc
    flat_store_dwordx4 v[74:75], v[108:111]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter8step3_qkE
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 148
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 148
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter8step3_qkE
    .symbol: _ZN5aiter8step3_qkE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 148
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

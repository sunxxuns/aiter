// Integration Step 1: QK MFMA
// Load Q[32,16] and K[32,16] from global memory
// Compute S = Q @ K^T using v_mfma_f32_32x32x16_fp8_fp8
// Output S[32,32] to global memory
//
// MFMA Thread mapping (64 threads):
// - Thread t provides A[M=t%32, K=(t/32)*8:(t/32)*8+8]
// - Thread t provides B[K=(t/32)*8:(t/32)*8+8, N=t%32]
// - Each thread loads 8 FP8 values for Q and 8 for K

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter17integrate_step1_qkE
.p2align 8
.type _ZN5aiter17integrate_step1_qkE,@function

_ZN5aiter17integrate_step1_qkE:
    // Kernarg: s[0:1] = kernarg_segment_ptr
    // Args at offset: 0=ptr_out, 8=ptr_Q, 16=ptr_K
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_out
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // ptr_K
    
    // Get thread ID (v0 has it from dispatch)
    v_and_b32_e32 v0, 63, v0               // tid = v0 & 63
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize accumulators (S = QK^T output) in VGPRs
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
    // Load Q for A operand
    // Q: 32 queries × 16 head_dim, stored row-major
    // Thread t needs Q[q=t%32, d=(t/32)*8:(t/32)*8+8]
    // Q address = ptr_Q + q * 16 + d_start
    // ========================================================================
    
    v_and_b32_e32 v1, 31, v0              // q = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // d_group = tid / 32
    v_lshlrev_b32_e32 v2, 3, v2           // d_start = d_group * 8
    
    v_lshlrev_b32_e32 v3, 4, v1           // q * 16 (row stride)
    v_add_u32_e32 v3, v3, v2              // + d_start
    
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 Q values (2 dwords = 8 FP8 bytes)
    flat_load_dwordx2 v[20:21], v[10:11]
    
    // ========================================================================
    // Load K for B operand
    // Thread t needs K[k=t%32, d=(t/32)*8:(t/32)*8+8]
    // K address = ptr_K + k * 16 + d_start
    // ========================================================================
    
    v_and_b32_e32 v1, 31, v0              // k = tid % 32
    v_lshrrev_b32_e32 v2, 5, v0           // d_group = tid / 32
    v_lshlrev_b32_e32 v2, 3, v2           // d_start = d_group * 8
    
    v_lshlrev_b32_e32 v3, 4, v1           // k * 16 (row stride)
    v_add_u32_e32 v3, v3, v2              // + d_start
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 K values
    flat_load_dwordx2 v[64:65], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Move Q to AGPRs for A operand
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    
    s_nop 7
    
    // ========================================================================
    // QK MFMA: S = Q @ K^T
    // A operand = Q (in a[0:1])
    // B operand = K (in v[64:65])
    // Accumulator in v[32:47]
    // ========================================================================
    
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 15
    s_nop 7
    
    // ========================================================================
    // Store S output with scatter pattern for interleaved M rows
    // M_base = (tid/32) * 4 (0 for threads 0-31, 4 for threads 32-63)
    // Each thread stores 16 values to rows M_base+0, M_base+8, M_base+16, M_base+24
    // with offsets +0,+1,+2,+3 within each group
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    v_and_b32_e32 v3, 31, v0              // N = tid % 32
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4 bytes
    v_lshrrev_b32_e32 v4, 5, v0           // M_base_idx = tid / 32
    v_lshlrev_b32_e32 v4, 2, v4           // M_base = M_base_idx * 4
    
    // Store v32-v35 to M_base + 0,1,2,3
    v_lshlrev_b32_e32 v5, 7, v4           // M_base * 128 (32 cols * 4 bytes)
    v_add_u32_e32 v5, v5, v3              // + N * 4
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v32
    flat_store_dword v[12:13], v33 offset:128
    flat_store_dword v[12:13], v34 offset:256
    flat_store_dword v[12:13], v35 offset:384
    
    // Store v36-v39 to M_base + 8,9,10,11
    v_add_u32_e32 v6, 8, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v36
    flat_store_dword v[12:13], v37 offset:128
    flat_store_dword v[12:13], v38 offset:256
    flat_store_dword v[12:13], v39 offset:384
    
    // Store v40-v43 to M_base + 16,17,18,19
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v40
    flat_store_dword v[12:13], v41 offset:128
    flat_store_dword v[12:13], v42 offset:256
    flat_store_dword v[12:13], v43 offset:384
    
    // Store v44-v47 to M_base + 24,25,26,27
    v_add_u32_e32 v6, 24, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v44
    flat_store_dword v[12:13], v45 offset:128
    flat_store_dword v[12:13], v46 offset:256
    flat_store_dword v[12:13], v47 offset:384
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter17integrate_step1_qkE, .-_ZN5aiter17integrate_step1_qkE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter17integrate_step1_qkE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
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
  - .name: _ZN5aiter17integrate_step1_qkE
    .symbol: _ZN5aiter17integrate_step1_qkE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
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
      - .name: ptr_Q
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_K
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

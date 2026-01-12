// Minimal QK MFMA test - Output raw v32-v47 for all threads
// Expected: 128.0 when Q=1, K=1

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13step3_raw_mfmaE
.p2align 8
.type _ZN5aiter13step3_raw_mfmaE,@function

_ZN5aiter13step3_raw_mfmaE:
    s_mov_b64 exec, -1
    
    // Load kernel args
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // Output [64*16] floats
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [32×128] FP8
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // Setup buffer descriptors
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    
    // ========================================================================
    // LOAD Q TO LDS (EXACT same as working kernel)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v0    // tid * 16
    
    s_mov_b32 m0, 0
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 1024
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 2048
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 3072
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K TO LDS (EXACT same as working kernel)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v0    // tid * 16
    
    s_mov_b32 m0, 4096
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 5120
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 6144
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 7168
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // QK MFMA (EXACT copy from working kernel)
    // ========================================================================
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
    
    // Read Q and K from LDS (EXACT same as working kernel)
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4
    v_add_u32_e32 v6, 4096, v5            // K base at LDS offset 4096
    
    // 8 QK MFMA passes: S^T = K @ Q^T (EXACT copy)
    // A operand = K, B operand = Q (swapped for transpose)
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \k_off, v6      // K offset (A operand)
        v_add_u32_e32 v8, \k_off, v5      // Q offset (B operand)
        ds_read_b64 v[20:21], v7          // Read K for A operand
        ds_read_b64 v[22:23], v8          // Read Q for B operand
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v20       // K → A
        v_accvgpr_write_b32 a1, v21
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
        s_nop 15
    .endr
    s_nop 15
    
    // ========================================================================
    // STORE RAW ACCUMULATOR VALUES
    // Output: 64 threads × 16 values = 1024 floats
    // Layout: out[tid * 16 + 0..15] = v32..v47
    // ========================================================================
    v_lshlrev_b32_e32 v1, 6, v0           // tid * 64 bytes (16 floats)
    v_mov_b32_e32 v2, s4
    v_mov_b32_e32 v3, s5
    v_add_co_u32_e32 v2, vcc, v1, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    
    // Store v32-v35 as dwordx4
    flat_store_dwordx4 v[2:3], v[32:35]
    v_add_co_u32_e32 v2, vcc, 16, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    
    // Store v36-v39 as dwordx4
    flat_store_dwordx4 v[2:3], v[36:39]
    v_add_co_u32_e32 v2, vcc, 16, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    
    // Store v40-v43 as dwordx4
    flat_store_dwordx4 v[2:3], v[40:43]
    v_add_co_u32_e32 v2, vcc, 16, v2
    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
    
    // Store v44-v47 as dwordx4
    flat_store_dwordx4 v[2:3], v[44:47]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13step3_raw_mfmaE
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 148
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 148
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter13step3_raw_mfmaE
    .symbol: _ZN5aiter13step3_raw_mfmaE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 148
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_out, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

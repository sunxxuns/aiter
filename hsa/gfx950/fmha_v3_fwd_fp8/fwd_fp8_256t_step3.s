// FP8 256T Step 3: QK MFMA Test
// 
// Purpose: Verify QK MFMA computation with 256 threads (wave 0 only computes)
// This is a direct adaptation of fwd_fp8_kloop.s with 256 thread launch
//
// LDS Layout:
//   [0:4095]    Q (32 rows × 128 cols, shared by all waves)
//   [4096:8191] K (32 rows × 128 cols, shared by all waves)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16fwd_fp8_256t_s3E
.p2align 8
.type _ZN5aiter16fwd_fp8_256t_s3E,@function

_ZN5aiter16fwd_fp8_256t_s3E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // THREAD DECOMPOSITION + EARLY MASK (BEFORE args load, like fwd_fp8_256t_min.s)
    // ========================================================================
    v_lshrrev_b32_e32 v1, 6, v0            // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0               // v0 = lane_id (0-63)
    
    // Mask to wave 0 immediately
    v_cmp_eq_u32_e32 vcc, 0, v1
    s_and_saveexec_b64 s[30:31], vcc       // Use s[30:31] like fwd_fp8_256t_min.s
    s_cbranch_execz END_KERNEL             // Waves 1-3 skip everything
    
    // ========================================================================
    // LOAD KERNEL ARGS (only wave 0 reaches here)
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q input [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K input [32×128] FP8
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // SETUP BUFFER DESCRIPTORS
    // ========================================================================
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // ========================================================================
    // LOAD Q TO LDS (WAVE 0 ONLY - 4 passes like 64T)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v0            // v1 = lane_id * 16
    
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
    
    // ========================================================================
    // LOAD K TO LDS (WAVE 0 ONLY - 4 passes like 64T)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v0            // v1 = lane_id * 16 (recalc)
    
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
    // QK MFMA (WAVE 0 - already masked above)
    // ========================================================================
    
    // Initialize S accumulator
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
    
    // Calculate LDS addresses (same as fwd_fp8_kloop.s)
    v_and_b32_e32 v2, 31, v0               // v2 = lane % 32
    v_lshrrev_b32_e32 v3, 5, v0            // v3 = lane / 32
    v_lshlrev_b32_e32 v5, 7, v2            // v5 = (lane % 32) * 128
    v_lshlrev_b32_e32 v4, 3, v3            // v4 = (lane / 32) * 8
    v_add_u32_e32 v5, v5, v4               // v5 = Q LDS base
    v_add_u32_e32 v6, 4096, v5             // v6 = K LDS base
    
    // 8 QK MFMA passes (same as fwd_fp8_kloop.s lines 177-188)
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \k_off, v6       // K offset
        v_add_u32_e32 v8, \k_off, v5       // Q offset
        ds_read_b64 v[20:21], v7           // Read K
        ds_read_b64 v[22:23], v8           // Read Q
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v20        // K → A
        v_accvgpr_write_b32 a1, v21
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
        s_nop 15
    .endr
    s_nop 15
    
    // ========================================================================
    // OUTPUT (LANE 0 ONLY within wave 0) - raw MFMA output
    // ========================================================================
    v_cmp_eq_u32_e32 vcc, 0, v0            // lane_id == 0
    s_and_saveexec_b64 s[24:25], vcc
    s_cbranch_execz END_KERNEL
    
    // Output raw S values from MFMA (v32-v47)
    v_mov_b32_e32 v7, 0
    buffer_store_dword v32, v7, s[4:7], 0 offen offset:0   // S[0]
    buffer_store_dword v33, v7, s[4:7], 0 offen offset:4   // S[1]
    buffer_store_dword v34, v7, s[4:7], 0 offen offset:8   // S[2]
    buffer_store_dword v35, v7, s[4:7], 0 offen offset:12  // S[3]
    
END_KERNEL:
    s_mov_b64 exec, -1
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter16fwd_fp8_256t_s3E, .-_ZN5aiter16fwd_fp8_256t_s3E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_256t_s3E
    .amdhsa_group_segment_fixed_size 12288
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 64
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
  - .name: _ZN5aiter16fwd_fp8_256t_s3E
    .symbol: _ZN5aiter16fwd_fp8_256t_s3E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 64
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

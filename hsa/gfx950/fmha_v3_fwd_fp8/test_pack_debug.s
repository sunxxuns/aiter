// Debug: Compare manual byte packing vs v_cvt_pk_fp8_f32
// Store both packing methods and compare

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13test_pack_dbgE
.p2align 8
.type _ZN5aiter13test_pack_dbgE,@function

_ZN5aiter13test_pack_dbgE:
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
    
    // A operand: P = 0.0625
    v_mov_b32_e32 v2, 0x3d800000
    v_mov_b32_e32 v4, 0
    v_cvt_pk_fp8_f32 v4, v2, v2
    v_lshlrev_b32_e32 v5, 16, v4
    v_or_b32_e32 v4, v4, v5
    v_mov_b32_e32 v5, v4
    v_accvgpr_write_b32 a0, v4
    v_accvgpr_write_b32 a1, v5
    
    // B operand METHOD 1: Using v_cvt_pk_fp8_f32
    // V = 0.03125 for all positions
    v_mov_b32_e32 v6, 0x3d000000          // 0.03125
    v_mov_b32_e32 v64, 0
    v_cvt_pk_fp8_f32 v64, v6, v6
    v_lshlrev_b32_e32 v65, 16, v64
    v_or_b32_e32 v64, v64, v65
    v_mov_b32_e32 v65, v64
    
    s_nop 7
    
    // MFMA 1: Using cvt_pk packed B
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    
    s_nop 7
    
    // Store output
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v3, 6, v0
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[32:35]
    
    // Now test METHOD 2: Manual packing
    // Simulate loading byte 24 (FP8 for 0.03125) and packing
    v_mov_b32_e32 v20, 24                 // Byte value
    v_lshlrev_b32_e32 v21, 8, v20
    v_or_b32_e32 v22, v20, v21            // [24, 24, 0, 0]
    v_lshlrev_b32_e32 v23, 16, v20
    v_or_b32_e32 v22, v22, v23            // [24, 24, 24, 0]
    v_lshlrev_b32_e32 v24, 24, v20
    v_or_b32_e32 v66, v22, v24            // [24, 24, 24, 24]
    v_mov_b32_e32 v67, v66
    
    // Reset accumulators
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
    
    s_nop 7
    
    // MFMA 2: Using manually packed B
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[66:67], v[48:63]
    
    s_nop 7
    
    // Store second output (offset by 1KB)
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[48:51]
    
    // Also store the packed values for inspection
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dword v[10:11], v64                  // cvt_pk packed
    flat_store_dword v[10:11], v66 offset:4        // manual packed
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter13test_pack_dbgE, .-_ZN5aiter13test_pack_dbgE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13test_pack_dbgE
    .amdhsa_group_segment_fixed_size 0
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
  - .name: _ZN5aiter13test_pack_dbgE
    .symbol: _ZN5aiter13test_pack_dbgE.kd
    .kernarg_segment_size: 16
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
...
.end_amdgpu_metadata

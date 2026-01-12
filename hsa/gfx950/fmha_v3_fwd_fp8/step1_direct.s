// Step 1: Direct wave offset computation without v_readfirstlane
// This avoids the cross-wave scalar issue

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter12step1_directE
.p2align 8
.type _ZN5aiter12step1_directE,@function

_ZN5aiter12step1_directE:
    s_mov_b64 exec, -1
    
    s_load_dwordx2 s[4:5], s[0:1], 0
    s_waitcnt lgkmcnt(0)
    
    // v0 = global thread ID (0-255)
    // Keep in VGPR for all calculations (don't use SGPR for wave-varying values!)
    
    v_mov_b32_e32 v16, v0                 // save original
    
    // Compute wave_id in VGPR (not SGPR!)
    v_lshrrev_b32_e32 v1, 6, v0           // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0              // v0 = lane within wave (0-63)
    
    // Create FP8 pair: [tid, tid+128]
    v_add_u32_e32 v3, 128, v16
    v_and_b32_e32 v3, 0xFF, v3
    v_lshlrev_b32_e32 v3, 8, v3
    v_and_b32_e32 v4, 0xFF, v16
    v_or_b32_e32 v3, v4, v3               // v3 = [tid+128, tid]
    
    // Compute write address (all in VGPRs)
    // Base formula for within-wave
    v_and_b32_e32 v10, 31, v0
    v_and_b32_e32 v11, 1, v10
    v_lshlrev_b32_e32 v11, 7, v11         // (lane & 1) * 0x80
    v_lshrrev_b32_e32 v12, 1, v10
    v_mul_u32_u24_e32 v12, 0x408, v12     // (lane >> 1) * 0x408
    v_lshrrev_b32_e32 v13, 5, v0
    v_lshlrev_b32_e32 v13, 4, v13         // (lane >> 5) * 16
    
    v_add_u32_e32 v14, v11, v12
    v_add_u32_e32 v14, v13, v14
    v_add_u32_e32 v14, 0x8200, v14        // + base
    
    // Wave offset (VGPR computation!)
    // (wave_id & 3) * 0x100 + (wave_id >> 2) * 0x8100
    v_and_b32_e32 v17, 3, v1              // wave_id & 3
    v_lshlrev_b32_e32 v17, 8, v17         // * 0x100
    v_add_u32_e32 v14, v17, v14
    v_lshrrev_b32_e32 v18, 2, v1          // wave_id >> 2
    v_mul_u32_u24_e32 v18, 0x8100, v18    // * 0x8100
    v_add_u32_e32 v14, v18, v14
    
    // Store wave_id for debug
    v_mov_b32_e32 v15, v1
    
    // Write FP8 pair
    ds_write_b16 v14, v3
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Read with transpose
    ds_read_b64_tr_b16 v[20:21], v14
    ds_read_b64_tr_b16 v[22:23], v14 offset:512
    s_waitcnt lgkmcnt(0)
    
    // Output
    v_lshlrev_b32_e32 v30, 5, v16
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_add_co_u32_e32 v40, vcc, v30, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    v_mov_b32_e32 v24, v3
    v_mov_b32_e32 v25, v14
    v_mov_b32_e32 v26, v15
    v_mov_b32_e32 v27, v16
    flat_store_dwordx4 v[40:41], v[20:23]
    v_add_co_u32_e32 v42, vcc, 16, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[24:27]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter12step1_directE
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 48
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 48
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter12step1_directE
    .symbol: _ZN5aiter12step1_directE.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 48
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

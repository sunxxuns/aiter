// Test 128Q - QK MFMA only, output S directly (no softmax, no PV)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16test_128q_qkonlyE
.p2align 8
.type _ZN5aiter16test_128q_qkonlyE,@function

_ZN5aiter16test_128q_qkonlyE:
    s_mov_b64 exec, -1
    
    // Args: O[128×32], Q[128×128], K[32×128]
    s_load_dwordx2 s[4:5], s[0:1], 0x00
    s_load_dwordx2 s[8:9], s[0:1], 0x08
    s_load_dwordx2 s[12:13], s[0:1], 0x10
    
    v_lshrrev_b32_e32 v61, 6, v0
    v_and_b32_e32 v60, 63, v0
    v_readfirstlane_b32 s28, v61
    
    s_waitcnt lgkmcnt(0)
    
    // Load Q to LDS (128×128 = 16KB)
    v_lshlrev_b32_e32 v1, 4, v0
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    .irp iter, 0, 4096, 8192, 12288
        .if \iter > 0
            v_add_u32_e32 v1, 4096, v1
            v_add_co_u32_e32 v10, vcc, 4096, v10
            v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
        .endif
        flat_load_dwordx4 v[20:23], v[10:11]
        s_waitcnt vmcnt(0)
        ds_write_b128 v1, v[20:23]
    .endr
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Load K to LDS at 16384 (32×128 = 4KB)
    v_lshlrev_b32_e32 v1, 4, v0
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v1, 16384, v1
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // MFMA row mapping
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v62, v3, v4
    
    v_mov_b32_e32 v63, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e32 v63, v63, v4, vcc
    
    // Q LDS addr: wave_id * 4096 + mfma_row * 128 + k_base
    s_mul_i32 s40, s28, 4096
    v_lshlrev_b32_e32 v70, 7, v62
    v_add_u32_e32 v70, v70, v63
    v_add_u32_e32 v70, s40, v70
    
    // K LDS addr: 16384 + mfma_row * 128 + k_base
    v_lshlrev_b32_e32 v71, 7, v62
    v_add_u32_e32 v71, v71, v63
    v_add_u32_e32 v71, 16384, v71
    
    // Clear S accumulators
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // 8 MFMAs for HD=128
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v71
        ds_read_b64 v[30:31], v72
        v_add_u32_e32 v73, \k_off, v70
        ds_read_b64 v[32:33], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    s_nop 15
    
    // Output S directly - use flat layout: wave_id * 4096 + lane * 64
    // Each wave outputs 64 lanes × 64 bytes = 4096 bytes
    // IMPORTANT: Don't clobber v[0:15] which hold MFMA results!
    s_mul_i32 s40, s28, 4096              // wave_id * 4096
    
    v_lshlrev_b32_e32 v50, 6, v60         // lane * 64 (use v50, not v1!)
    v_add_u32_e32 v50, s40, v50
    
    v_mov_b32_e32 v16, s4
    v_mov_b32_e32 v17, s5
    v_add_co_u32_e32 v16, vcc, v50, v16
    v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
    
    // Store v[0:15] (16 floats = 64 bytes per lane)
    flat_store_dwordx4 v[16:17], v[0:3]
    v_add_co_u32_e32 v18, vcc, 16, v16
    v_addc_co_u32_e32 v19, vcc, 0, v17, vcc
    flat_store_dwordx4 v[18:19], v[4:7]
    v_add_co_u32_e32 v18, vcc, 32, v16
    v_addc_co_u32_e32 v19, vcc, 0, v17, vcc
    flat_store_dwordx4 v[18:19], v[8:11]
    v_add_co_u32_e32 v18, vcc, 48, v16
    v_addc_co_u32_e32 v19, vcc, 0, v17, vcc
    flat_store_dwordx4 v[18:19], v[12:15]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16test_128q_qkonlyE
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter16test_128q_qkonlyE
    .symbol: _ZN5aiter16test_128q_qkonlyE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 80
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

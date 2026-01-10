// Full Attention Test v2: Use scatter stores based on actual MFMA output positions
// Skips permlane swap, stores directly to correct output positions

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter17test_full_attn_v2E
.p2align 8
.type _ZN5aiter17test_full_attn_v2E,@function

.set LDS_P_OFFSET, 0

_ZN5aiter17test_full_attn_v2E:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_out
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_V
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize accumulators
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
    
    // ========================================================================
    // Generate P = 1/16 and write to LDS
    // ========================================================================
    
    v_mov_b32_e32 v2, 0x3d800000          // 0.0625 = 1/16
    v_mov_b32_e32 v3, 0
    v_cvt_pk_fp8_f32 v3, v2, v2
    
    v_and_b32_e32 v4, 31, v0
    v_add_u32_e32 v5, LDS_P_OFFSET, v4
    
    // Write P to all 32 Q rows
    ds_write_b8 v5, v3
    ds_write_b8 v5, v3 offset:32
    ds_write_b8 v5, v3 offset:64
    ds_write_b8 v5, v3 offset:96
    ds_write_b8 v5, v3 offset:128
    ds_write_b8 v5, v3 offset:160
    ds_write_b8 v5, v3 offset:192
    ds_write_b8 v5, v3 offset:224
    ds_write_b8 v5, v3 offset:256
    ds_write_b8 v5, v3 offset:288
    ds_write_b8 v5, v3 offset:320
    ds_write_b8 v5, v3 offset:352
    ds_write_b8 v5, v3 offset:384
    ds_write_b8 v5, v3 offset:416
    ds_write_b8 v5, v3 offset:448
    ds_write_b8 v5, v3 offset:480
    ds_write_b8 v5, v3 offset:512
    ds_write_b8 v5, v3 offset:544
    ds_write_b8 v5, v3 offset:576
    ds_write_b8 v5, v3 offset:608
    ds_write_b8 v5, v3 offset:640
    ds_write_b8 v5, v3 offset:672
    ds_write_b8 v5, v3 offset:704
    ds_write_b8 v5, v3 offset:736
    ds_write_b8 v5, v3 offset:768
    ds_write_b8 v5, v3 offset:800
    ds_write_b8 v5, v3 offset:832
    ds_write_b8 v5, v3 offset:864
    ds_write_b8 v5, v3 offset:896
    ds_write_b8 v5, v3 offset:928
    ds_write_b8 v5, v3 offset:960
    ds_write_b8 v5, v3 offset:992
    
    s_barrier
    
    // ========================================================================
    // Read P for PV MFMA A operand
    // ========================================================================
    
    v_and_b32_e32 v6, 31, v0
    v_lshrrev_b32_e32 v7, 5, v0
    v_lshlrev_b32_e32 v7, 3, v7
    v_lshlrev_b32_e32 v6, 5, v6
    v_add_u32_e32 v6, v6, v7
    v_add_u32_e32 v6, LDS_P_OFFSET, v6
    
    ds_read_b64 v[32:33], v6
    
    s_waitcnt lgkmcnt(0)
    
    v_accvgpr_write_b32 a0, v32
    v_accvgpr_write_b32 a1, v33
    
    // ========================================================================
    // Load V for B operand: V[k,d] = d/32 from global memory
    // V is stored row-major: V[k,d] at ptr_V + k*32 + d
    // Thread t needs V[K_range, D=t%32]
    // ========================================================================
    
    v_and_b32_e32 v8, 31, v0              // D = tid % 32
    v_lshrrev_b32_e32 v9, 5, v0           // K_group
    v_lshlrev_b32_e32 v9, 8, v9           // K_group * 8 * 32 = K_start * stride
    
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v8, v10    // + D
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v9, v10    // + K_start * stride
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 V values with stride 32
    flat_load_ubyte v34, v[10:11]
    flat_load_ubyte v35, v[10:11] offset:32
    flat_load_ubyte v36, v[10:11] offset:64
    flat_load_ubyte v37, v[10:11] offset:96
    flat_load_ubyte v38, v[10:11] offset:128
    flat_load_ubyte v39, v[10:11] offset:160
    flat_load_ubyte v40, v[10:11] offset:192
    flat_load_ubyte v41, v[10:11] offset:224
    
    s_waitcnt vmcnt(0)
    
    // Pack V
    v_lshlrev_b32_e32 v35, 8, v35
    v_or_b32_e32 v34, v34, v35
    v_lshlrev_b32_e32 v36, 16, v36
    v_or_b32_e32 v34, v34, v36
    v_lshlrev_b32_e32 v37, 24, v37
    v_or_b32_e32 v64, v34, v37
    
    v_lshlrev_b32_e32 v39, 8, v39
    v_or_b32_e32 v38, v38, v39
    v_lshlrev_b32_e32 v40, 16, v40
    v_or_b32_e32 v38, v38, v40
    v_lshlrev_b32_e32 v41, 24, v41
    v_or_b32_e32 v65, v38, v41
    
    s_nop 7
    
    // ========================================================================
    // PV MFMA
    // ========================================================================
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    s_nop 7
    
    // ========================================================================
    // Scatter store based on actual MFMA output M-row positions
    // Thread t owns 16 output values at N=t%32
    // M rows are interleaved:
    //   Threads 0-31: M = 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
    //   Threads 32-63: M = 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31
    //
    // Output address for C[M, N]: ptr_out + M * 32 * 4 + N * 4
    //                           = ptr_out + M * 128 + N * 4
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    // N = tid % 32
    v_and_b32_e32 v3, 31, v0
    v_lshlrev_b32_e32 v3, 2, v3           // N * 4
    
    // M_base = (tid / 32) * 4 (either 0 or 4)
    v_lshrrev_b32_e32 v4, 5, v0           // tid / 32
    v_lshlrev_b32_e32 v4, 2, v4           // * 4
    
    // Store v48-v51 to M_base + 0,1,2,3
    // M_row 0: v48 at ptr + (M_base+0)*128 + N*4
    v_lshlrev_b32_e32 v5, 7, v4           // (M_base) * 128
    v_add_u32_e32 v5, v5, v3              // + N * 4
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v48                 // M_base + 0
    flat_store_dword v[12:13], v49 offset:128     // M_base + 1
    flat_store_dword v[12:13], v50 offset:256     // M_base + 2
    flat_store_dword v[12:13], v51 offset:384     // M_base + 3
    
    // Store v52-v55 to M_base + 8,9,10,11
    // For threads 0-31: M = 8,9,10,11
    // For threads 32-63: M = 12,13,14,15
    v_add_u32_e32 v6, 8, v4               // M_base + 8
    v_lshlrev_b32_e32 v5, 7, v6           // * 128
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v52                 // M_base + 8
    flat_store_dword v[12:13], v53 offset:128     // M_base + 9
    flat_store_dword v[12:13], v54 offset:256     // M_base + 10
    flat_store_dword v[12:13], v55 offset:384     // M_base + 11
    
    // Store v56-v59 to M_base + 16,17,18,19 or 20,21,22,23
    v_add_u32_e32 v6, 16, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v56
    flat_store_dword v[12:13], v57 offset:128
    flat_store_dword v[12:13], v58 offset:256
    flat_store_dword v[12:13], v59 offset:384
    
    // Store v60-v63 to M_base + 24,25,26,27 or 28,29,30,31
    v_add_u32_e32 v6, 24, v4
    v_lshlrev_b32_e32 v5, 7, v6
    v_add_u32_e32 v5, v5, v3
    v_add_co_u32_e32 v12, vcc, v5, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dword v[12:13], v60
    flat_store_dword v[12:13], v61 offset:128
    flat_store_dword v[12:13], v62 offset:256
    flat_store_dword v[12:13], v63 offset:384
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter17test_full_attn_v2E, .-_ZN5aiter17test_full_attn_v2E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter17test_full_attn_v2E
    .amdhsa_group_segment_fixed_size 4096
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
  - .name: _ZN5aiter17test_full_attn_v2E
    .symbol: _ZN5aiter17test_full_attn_v2E.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 4096
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
      - .name: ptr_V
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

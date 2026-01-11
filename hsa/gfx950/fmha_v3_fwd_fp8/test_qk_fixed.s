// QK MFMA with buffer_load→LDS and CORRECT store pattern
// Based on decoded MFMA output layout:
//   row = ((vreg - 32) % 4) + (tid // 32) * 4 + ((vreg - 32) // 4) * 8
//   col = tid % 32

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13test_qk_fixedE
.p2align 8
.type _ZN5aiter13test_qk_fixedE,@function

_ZN5aiter13test_qk_fixedE:
    s_mov_b64 exec, -1
    
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // S output
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // Load Q to LDS
    s_mov_b32 s10, 1024
    s_mov_b32 s11, 0x20000
    v_lshlrev_b32_e32 v1, 4, v0
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v1, s[8:11], 0 offen lds
    
    // Load K to LDS
    s_mov_b32 s14, 1024
    s_mov_b32 s15, 0x20000
    v_lshlrev_b32_e32 v1, 4, v0
    s_mov_b32 m0, 1024
    buffer_load_dwordx4 v1, s[12:15], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // Initialize accumulators
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
    
    // Read Q/K for MFMA
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v4, 3, v3
    v_lshlrev_b32_e32 v5, 5, v2
    v_add_u32_e32 v5, v5, v4
    
    ds_read_b64 v[20:21], v5
    v_add_u32_e32 v6, 1024, v5
    ds_read_b64 v[22:23], v6
    
    s_waitcnt lgkmcnt(0)
    
    // MFMA pass 1
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // MFMA pass 2
    v_add_u32_e32 v5, 16, v5
    v_add_u32_e32 v6, 16, v6
    ds_read_b64 v[20:21], v5
    ds_read_b64 v[22:23], v6
    s_waitcnt lgkmcnt(0)
    
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    s_nop 15
    s_nop 15
    
    // ========================================================================
    // Store with CORRECT pattern
    // For vreg v and thread tid:
    //   row = ((v-32) % 4) + (tid // 32) * 4 + ((v-32) // 4) * 8
    //   col = tid % 32
    //   addr = ptr_S + (row * 32 + col) * 4
    // ========================================================================
    
    // Pre-calculate common terms
    v_and_b32_e32 v2, 31, v0               // col = tid % 32
    v_lshrrev_b32_e32 v3, 5, v0            // tid // 32 (0 or 1)
    v_lshlrev_b32_e32 v3, 2, v3            // (tid // 32) * 4
    
    // Macro: store vreg to S[row, col]
    // row = row_mod4 + v3 + row_8_group * 8
    .macro STORE_VREG vreg, row_mod4, row_8_group
        v_mov_b32_e32 v7, \row_mod4
        v_add_u32_e32 v7, v7, v3                    // row_mod4 + (tid//32)*4
        v_add_u32_e32 v7, \row_8_group * 8, v7     // + row_8_group * 8
        v_lshlrev_b32_e32 v7, 5, v7                 // row * 32
        v_add_u32_e32 v7, v7, v2                    // row * 32 + col
        v_lshlrev_b32_e32 v7, 2, v7                 // byte offset
        v_mov_b32_e32 v10, s4
        v_mov_b32_e32 v11, s5
        v_add_co_u32_e32 v10, vcc, v7, v10
        v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
        flat_store_dword v[10:11], \vreg
    .endm
    
    // v32-v35: row_8_group=0, row_mod4 = 0,1,2,3
    STORE_VREG v32, 0, 0
    STORE_VREG v33, 1, 0
    STORE_VREG v34, 2, 0
    STORE_VREG v35, 3, 0
    
    // v36-v39: row_8_group=1, row_mod4 = 0,1,2,3
    STORE_VREG v36, 0, 1
    STORE_VREG v37, 1, 1
    STORE_VREG v38, 2, 1
    STORE_VREG v39, 3, 1
    
    // v40-v43: row_8_group=2, row_mod4 = 0,1,2,3
    STORE_VREG v40, 0, 2
    STORE_VREG v41, 1, 2
    STORE_VREG v42, 2, 2
    STORE_VREG v43, 3, 2
    
    // v44-v47: row_8_group=3, row_mod4 = 0,1,2,3
    STORE_VREG v44, 0, 3
    STORE_VREG v45, 1, 3
    STORE_VREG v46, 2, 3
    STORE_VREG v47, 3, 3
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13test_qk_fixedE
    .amdhsa_group_segment_fixed_size 2048
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 52
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 48
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
  - .name: _ZN5aiter13test_qk_fixedE
    .symbol: _ZN5aiter13test_qk_fixedE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 2048
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 52
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_S, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

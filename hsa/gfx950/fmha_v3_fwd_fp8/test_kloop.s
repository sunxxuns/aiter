// FP8 Flash Attention with K-tile Loop
// Tests: Load K from multiple tiles using scalar offset
// Simplified: Only tests K loading + accumulation, no full attention

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter10test_kloopE
.p2align 8
.type _ZN5aiter10test_kloopE,@function

_ZN5aiter10test_kloopE:
    s_mov_b64 exec, -1
    
    // Load kernel args
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [32×32] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // K input [seq×128] FP8
    s_load_dword s16, s[0:1], 0x10         // seq_len
    
    v_and_b32_e32 v0, 63, v0
    s_waitcnt lgkmcnt(0)
    
    // Setup K buffer descriptor
    s_mov_b32 s10, 0x100000                 // 1MB size
    s_mov_b32 s11, 0x20000                  // offen flags
    
    // Calculate number of tiles: (seq_len + 31) / 32
    s_add_i32 s17, s16, 31
    s_lshr_b32 s17, s17, 5                  // s17 = num_tiles
    
    // K-tile offset tracking
    s_mov_b32 s18, 0                        // k_offset = 0
    s_mov_b32 s19, 16384                    // k_stride = 32 * 128 * 4 bytes per F32 tile
    
    // Loop counter
    s_mov_b32 s20, 0                        // tile_idx = 0
    
    // Initialize accumulator
    v_mov_b32_e32 v10, 0                    // Sum of first elements across tiles
    
    // Thread's byte offset within tile
    v_lshlrev_b32_e32 v1, 2, v0             // v1 = tid * 4 (for F32 test data)
    
K_LOOP:
    // Load one element from current K tile
    buffer_load_dword v2, v1, s[8:11], s18 offen
    s_waitcnt vmcnt(0)
    
    // Accumulate
    v_add_f32_e32 v10, v10, v2
    
    // Advance offset
    s_add_i32 s18, s18, s19                 // k_offset += k_stride
    
    // Loop control
    s_add_i32 s20, s20, 1                   // tile_idx++
    s_cmp_lt_i32 s20, s17                   // if tile_idx < num_tiles
    s_cbranch_scc1 K_LOOP
    
    // Store result
    v_mov_b32_e32 v6, s4
    v_mov_b32_e32 v7, s5
    v_add_co_u32_e32 v6, vcc, v1, v6
    v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
    flat_store_dword v[6:7], v10
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter10test_kloopE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 12
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 12
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
  - .name: _ZN5aiter10test_kloopE
    .symbol: _ZN5aiter10test_kloopE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 12
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 16, .value_kind: by_value}
...
.end_amdgpu_metadata

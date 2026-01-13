// Minimal TR_B8 experiment - understand exactly what it does
// Fill LDS with known pattern, read with TR_B8, dump output

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter9exp_tr_b8E
.p2align 8
.type _ZN5aiter9exp_tr_b8E,@function

_ZN5aiter9exp_tr_b8E:
    s_mov_b64 exec, -1
    
    // Args: output (dump buffer), base_addr (scalar)
    s_load_dwordx2 s[4:5], s[0:1], 0     // output ptr
    s_load_dword s6, s[0:1], 8           // base_addr for TR_B8
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0   // lane_id
    
    // ========================================================================
    // Fill LDS[0:511] with pattern: LDS[i] = i (byte value = address)
    // ========================================================================
    
    // Each of 64 lanes writes 8 bytes
    v_lshlrev_b32_e32 v1, 3, v0          // lane * 8
    
    // Build 8 consecutive bytes: [lane*8, lane*8+1, ..., lane*8+7]
    // Pack into v2, v3 (2 dwords = 8 bytes)
    v_lshlrev_b32_e32 v2, 3, v0          // base = lane * 8
    v_add_u32_e32 v3, 1, v2
    v_lshlrev_b32_e32 v3, 8, v3
    v_or_b32_e32 v2, v2, v3              // bytes [0,1]
    
    v_lshlrev_b32_e32 v4, 3, v0
    v_add_u32_e32 v4, 2, v4
    v_lshlrev_b32_e32 v4, 16, v4
    v_or_b32_e32 v2, v2, v4              // bytes [0,1,2]
    
    v_lshlrev_b32_e32 v4, 3, v0
    v_add_u32_e32 v4, 3, v4
    v_lshlrev_b32_e32 v4, 24, v4
    v_or_b32_e32 v2, v2, v4              // bytes [0,1,2,3]
    
    // Second dword
    v_lshlrev_b32_e32 v3, 3, v0
    v_add_u32_e32 v3, 4, v3              // base+4
    v_mov_b32_e32 v4, v3
    v_add_u32_e32 v5, 1, v4
    v_lshlrev_b32_e32 v5, 8, v5
    v_or_b32_e32 v3, v3, v5
    v_add_u32_e32 v5, 2, v4
    v_lshlrev_b32_e32 v5, 16, v5
    v_or_b32_e32 v3, v3, v5
    v_add_u32_e32 v5, 3, v4
    v_lshlrev_b32_e32 v5, 24, v5
    v_or_b32_e32 v3, v3, v5              // bytes [4,5,6,7]
    
    ds_write_b64 v1, v[2:3]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // Read with TR_B8 from base_addr (passed as argument)
    // ========================================================================
    
    v_mov_b32_e32 v10, s6                // base_addr
    ds_read_b64_tr_b8 v[20:21], v10
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // Dump results: each lane writes its v[20:21] to output
    // ========================================================================
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 3, v60        // lane * 8 (each lane writes 8 bytes)
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx2 v[40:41], v[20:21]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter9exp_tr_b8E
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter9exp_tr_b8E
    .symbol: _ZN5aiter9exp_tr_b8E.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: base_addr, .size: 4, .offset: 8, .value_kind: by_value}
...
.end_amdgpu_metadata

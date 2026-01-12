// Step 2 Flat: Use flat_load instead of buffer_load
// This avoids buffer descriptor complexity

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter10step2_flatE
.p2align 8
.type _ZN5aiter10step2_flatE,@function

_ZN5aiter10step2_flatE:
    s_mov_b64 exec, -1
    
    // Args: [output_ptr, input_ptr]
    s_load_dwordx4 s[4:7], s[0:1], 0
    s_waitcnt lgkmcnt(0)
    
    // v0 = thread ID (0-255)
    v_mov_b32_e32 v60, v0
    
    // Each thread loads 16 bytes from input[tid*16]
    v_lshlrev_b32_e32 v10, 4, v0          // global offset = tid * 16
    v_mov_b32_e32 v12, s6                 // input base lo
    v_mov_b32_e32 v13, s7                 // input base hi
    v_add_co_u32_e32 v12, vcc, v10, v12
    v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
    
    // Load 16 bytes from global
    flat_load_dwordx4 v[20:23], v[12:13]
    s_waitcnt vmcnt(0)
    
    // ========================================================================
    // Write to LDS at SWIZZLED address (matching BF16's m0 pattern)
    // ========================================================================
    // BF16 m0 formula: base + (offset from global load formula)
    // We need to compute the LDS write address using the same swizzle
    
    // Simplified: just use the swizzle formula directly
    // LDS write addr = 0x8200 + swizzle_offset(tid)
    // where swizzle_offset uses BF16's pattern
    
    v_and_b32_e32 v14, 63, v0             // lane within wave
    v_and_b32_e32 v15, 31, v14
    v_and_b32_e32 v16, 1, v15
    v_lshlrev_b32_e32 v16, 7, v16         // (lane & 1) * 0x80
    v_lshrrev_b32_e32 v17, 1, v15
    v_mul_u32_u24_e32 v17, 0x408, v17     // (lane >> 1) * 0x408
    v_lshrrev_b32_e32 v18, 5, v14
    v_lshlrev_b32_e32 v18, 4, v18         // (lane >> 5) * 16
    
    v_add_u32_e32 v19, v16, v17
    v_add_u32_e32 v19, v18, v19
    v_add_u32_e32 v19, 0x8200, v19        // + base offset
    
    // Add wave offset (VGPR approach)
    v_lshrrev_b32_e32 v1, 6, v0           // wave_id
    v_and_b32_e32 v2, 3, v1
    v_lshlrev_b32_e32 v2, 8, v2           // (wave_id & 3) * 0x100
    v_add_u32_e32 v19, v2, v19
    v_lshrrev_b32_e32 v2, 2, v1
    v_mul_u32_u24_e32 v2, 0x8100, v2      // (wave_id >> 2) * 0x8100
    v_add_u32_e32 v19, v2, v19
    
    // Write to swizzled LDS address
    ds_write_b128 v19, v[20:23]
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Now test transpose read from LDS
    // Use BF16's swizzle formula for read base
    v_and_b32_e32 v30, 63, v0             // lane within wave
    v_and_b32_e32 v31, 31, v30
    v_and_b32_e32 v32, 1, v31
    v_lshlrev_b32_e32 v32, 7, v32         // (lane & 1) * 0x80
    v_lshrrev_b32_e32 v33, 1, v31
    v_mul_u32_u24_e32 v33, 0x408, v33     // (lane >> 1) * 0x408
    v_lshrrev_b32_e32 v34, 5, v30
    v_lshlrev_b32_e32 v34, 4, v34         // (lane >> 5) * 16
    
    v_add_u32_e32 v35, v32, v33
    v_add_u32_e32 v35, v34, v35
    v_add_u32_e32 v35, 0x8200, v35        // Add base offset (same as write)
    
    // Add wave offset
    v_lshrrev_b32_e32 v36, 6, v60         // wave_id
    v_and_b32_e32 v37, 3, v36
    v_lshlrev_b32_e32 v37, 8, v37         // (wave_id & 3) * 0x100
    v_add_u32_e32 v35, v37, v35
    v_lshrrev_b32_e32 v37, 2, v36
    v_mul_u32_u24_e32 v37, 0x8100, v37    // (wave_id >> 2) * 0x8100
    v_add_u32_e32 v35, v37, v35
    
    // Try transpose read
    ds_read_b64_tr_b16 v[40:41], v35
    s_waitcnt lgkmcnt(0)
    
    // Also do simple read for comparison
    ds_read_b64 v[42:43], v10
    s_waitcnt lgkmcnt(0)
    
    // Output: [tr_read, simple_read, tid, addr]
    v_lshlrev_b32_e32 v50, 5, v60
    v_mov_b32_e32 v52, s4
    v_mov_b32_e32 v53, s5
    v_add_co_u32_e32 v52, vcc, v50, v52
    v_addc_co_u32_e32 v53, vcc, 0, v53, vcc
    
    v_mov_b32_e32 v44, v60                // tid
    v_mov_b32_e32 v45, v35                // transpose read addr
    v_mov_b32_e32 v46, v10                // simple read addr
    v_mov_b32_e32 v47, 0
    
    flat_store_dwordx4 v[52:53], v[40:43]
    v_add_co_u32_e32 v54, vcc, 16, v52
    v_addc_co_u32_e32 v55, vcc, 0, v53, vcc
    flat_store_dwordx4 v[54:55], v[44:47]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter10step2_flatE
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
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
  - .name: _ZN5aiter10step2_flatE
    .symbol: _ZN5aiter10step2_flatE.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: input, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

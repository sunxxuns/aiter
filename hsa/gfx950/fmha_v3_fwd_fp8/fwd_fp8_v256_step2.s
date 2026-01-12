// FP8 Flash Attention 256T - Step 2: Q Load Verification
// Load Q to LDS, read back, output to verify correctness
//
// Each wave handles 8 Q-rows
// Wave 0: rows 0-7, Wave 1: rows 8-15, Wave 2: rows 16-23, Wave 3: rows 24-31

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16fwd_fp8_v256_s2E
.p2align 8
.type _ZN5aiter16fwd_fp8_v256_s2E,@function

_ZN5aiter16fwd_fp8_v256_s2E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // THREAD ID: 256 threads = 4 waves × 64 lanes
    // ========================================================================
    v_lshrrev_b32_e32 v1, 6, v0           // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0              // v0 = lane_id (0-63)
    
    // ========================================================================
    // SAVE WORKGROUP ID & LOAD ARGS
    // ========================================================================
    s_mov_b32 s28, s2
    
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // COMPUTE Q-TILE OFFSETS
    // ========================================================================
    s_lshl_b32 s29, s28, 12               // Q offset = tile * 4096
    s_lshl_b32 s30, s28, 14               // O offset = tile * 16384
    
    s_add_u32 s8, s8, s29
    s_addc_u32 s9, s9, 0
    s_add_u32 s4, s4, s30
    s_addc_u32 s5, s5, 0
    
    // ========================================================================
    // BUFFER DESCRIPTORS
    // ========================================================================
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // ========================================================================
    // LOAD Q TO LDS (all 256 threads cooperate)
    // ========================================================================
    // Q tile: 32 rows × 128 cols × 1 byte = 4096 bytes
    // 256 threads × 16 bytes/thread = 4096 bytes ✓
    
    v_lshlrev_b32_e32 v2, 6, v1           // wave * 64
    v_add_u32_e32 v2, v0, v2              // thread_id (0-255)
    v_lshlrev_b32_e32 v2, 4, v2           // thread_id * 16 (LDS offset)
    
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v2, s[8:11], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // READ Q FROM LDS (each lane reads one row's worth for its position)
    // ========================================================================
    // Each wave handles 8 Q-rows
    // Each lane within wave handles specific columns
    // MFMA 32x32x16 expects:
    //   - 64 lanes produce 32x32 output
    //   - Each lane contributes to multiple output elements
    //
    // For step2: just read some Q values and output to verify load worked
    
    // Simple mapping: thread_id maps to row = tid/4, col = (tid%4)*32 + (lane%32)
    // This gives us 64 rows if we had that many, but we only have 32
    // So: row = tid % 32, col_group = tid / 32
    
    v_lshlrev_b32_e32 v3, 6, v1           // wave * 64
    v_add_u32_e32 v3, v0, v3              // thread_id (0-255)
    
    v_and_b32_e32 v4, 31, v3              // row = tid % 32 (0-31)
    v_lshrrev_b32_e32 v5, 5, v3           // col_group = tid / 32 (0-7)
    
    // LDS Q offset: row * 128 + col_group * 16
    // We read 16 consecutive bytes (16 FP8 values) per thread
    v_lshlrev_b32_e32 v6, 7, v4           // row * 128
    v_lshlrev_b32_e32 v7, 4, v5           // col_group * 16
    v_add_u32_e32 v6, v6, v7              // LDS offset
    
    // Read 4 bytes (4 FP8 values) - simpler than 16
    ds_read_b32 v8, v6
    s_waitcnt lgkmcnt(0)
    
    // v8 now contains 4 FP8 values packed
    // For verification, just output the raw packed value as F32 bits
    // (This tests the LDS read works, not FP8 conversion)
    
    // ========================================================================
    // STORE OUTPUT
    // ========================================================================
    // O offset: thread_id * 4 bytes
    v_lshlrev_b32_e32 v9, 2, v3           // tid * 4
    
    // Store the packed FP8 bytes as-is (for debugging)
    buffer_store_dword v8, v9, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_v256_s2E
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 16
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
  - .name: _ZN5aiter16fwd_fp8_v256_s2E
    .symbol: _ZN5aiter16fwd_fp8_v256_s2E.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 16
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

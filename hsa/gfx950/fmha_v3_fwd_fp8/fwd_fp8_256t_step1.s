// FP8 256T Step 1: Wave-Aware Q Load Test
// 
// Purpose: Verify each wave loads its 8 Q rows to correct LDS region
// 
// LDS Layout (4KB total for Q):
//   Wave 0: Q rows 0-7   at LDS[0:1023]
//   Wave 1: Q rows 8-15  at LDS[1024:2047]
//   Wave 2: Q rows 16-23 at LDS[2048:3071]
//   Wave 3: Q rows 24-31 at LDS[3072:4095]
//
// Output: Each thread outputs (wave_id, lane_id, Q_value) as F32
//         O[thread_id*4 + 0] = wave_id (as float)
//         O[thread_id*4 + 1] = lane_id (as float)
//         O[thread_id*4 + 2] = Q[0] from this wave's LDS (first byte)
//         O[thread_id*4 + 3] = Q[127] from this wave's LDS (last byte of row 0)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16fwd_fp8_256t_s1E
.p2align 8
.type _ZN5aiter16fwd_fp8_256t_s1E,@function

_ZN5aiter16fwd_fp8_256t_s1E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q input [32×128] FP8
    
    // ========================================================================
    // THREAD DECOMPOSITION (256 threads = 4 waves × 64 lanes)
    // ========================================================================
    // v0 initially contains thread_id (0-255)
    v_mov_b32_e32 v10, v0                  // Save original thread_id
    v_lshrrev_b32_e32 v1, 6, v0            // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0               // v0 = lane_id (0-63)
    
    // Save wave_id to scalar for later use
    v_readfirstlane_b32 s20, v1            // s20 = wave_id (uniform per wave)
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // SETUP BUFFER DESCRIPTORS
    // ========================================================================
    // Q descriptor at s[8:11]
    s_mov_b32 s10, -1                      // size = max
    s_mov_b32 s11, 0x20000                 // flags (offen mode)
    
    // O descriptor at s[4:7]  
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // ========================================================================
    // WAVE-AWARE Q LOAD TO LDS
    // ========================================================================
    // Each wave loads its 8 Q rows (8 × 128 = 1024 bytes)
    // Wave 0: rows 0-7   → LDS[0:1023]     from Q[0:1023]
    // Wave 1: rows 8-15  → LDS[1024:2047]  from Q[1024:2047]
    // Wave 2: rows 16-23 → LDS[2048:3071]  from Q[2048:3071]
    // Wave 3: rows 24-31 → LDS[3072:4095]  from Q[3072:4095]
    
    // Calculate per-thread offset within wave's Q region
    // 64 threads × 16 bytes/thread = 1024 bytes = one wave's Q data
    v_lshlrev_b32_e32 v2, 4, v0            // v2 = lane_id * 16 (offset within 1KB)
    
    // Calculate wave's LDS destination
    s_mul_i32 s21, s20, 1024               // s21 = wave_id * 1024 (LDS base for wave)
    s_mov_b32 m0, s21                      // m0 = LDS destination base
    
    // Calculate wave's Q source offset
    // Q source = wave_id * 1024 + lane_id * 16
    v_add_u32_e32 v3, s21, v2              // v3 = wave_Q_base + lane_offset (for source)
    
    // Load Q to LDS (each lane loads 16 bytes)
    // buffer_load with offen lds: loads to LDS[m0 + v2]
    buffer_load_dwordx4 v2, s[8:11], s21 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // READ BACK FROM LDS AND OUTPUT VERIFICATION DATA
    // ========================================================================
    // Each thread outputs 4 values:
    // [0] = wave_id, [1] = lane_id, [2] = Q[0] from wave's LDS, [3] = Q[127]
    
    // Read Q[0] from this wave's LDS region (first byte of row 0)
    v_mov_b32_e32 v4, s21                  // v4 = wave's LDS base
    ds_read_u8 v5, v4                      // v5 = Q[0] as u8
    
    // Read Q[127] from this wave's LDS region (last byte of row 0)  
    v_add_u32_e32 v4, 127, v4              // v4 = wave_LDS_base + 127
    ds_read_u8 v6, v4                      // v6 = Q[127] as u8
    
    s_waitcnt lgkmcnt(0)
    
    // Convert to F32 for output
    v_cvt_f32_u32_e32 v20, v1              // v20 = wave_id as F32
    v_cvt_f32_u32_e32 v21, v0              // v21 = lane_id as F32
    v_cvt_f32_u32_e32 v22, v5              // v22 = Q[0] as F32 (FP8 byte value)
    v_cvt_f32_u32_e32 v23, v6              // v23 = Q[127] as F32 (FP8 byte value)
    
    // ========================================================================
    // STORE OUTPUT
    // ========================================================================
    // Output offset = thread_id * 16 bytes (4 floats)
    v_lshlrev_b32_e32 v7, 4, v10           // v7 = thread_id * 16
    
    buffer_store_dword v20, v7, s[4:7], 0 offen           // O[tid*4+0] = wave_id
    buffer_store_dword v21, v7, s[4:7], 0 offen offset:4  // O[tid*4+1] = lane_id  
    buffer_store_dword v22, v7, s[4:7], 0 offen offset:8  // O[tid*4+2] = Q[0]
    buffer_store_dword v23, v7, s[4:7], 0 offen offset:12 // O[tid*4+3] = Q[127]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter16fwd_fp8_256t_s1E, .-_ZN5aiter16fwd_fp8_256t_s1E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_256t_s1E
    .amdhsa_group_segment_fixed_size 4096
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 32
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 32
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
  - .name: _ZN5aiter16fwd_fp8_256t_s1E
    .symbol: _ZN5aiter16fwd_fp8_256t_s1E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 32
    .agpr_count: 0
    .max_flat_workgroup_size: 256
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

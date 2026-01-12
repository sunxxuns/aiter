// FP8 256T Step 2: Shared K Load Test
// 
// Purpose: Verify all 256 threads cooperatively load K to shared LDS region
// 
// LDS Layout:
//   [0:4095]    Q (per-wave regions, from Step 1)
//   [4096:8191] K (shared by all waves)
//
// Test: 
//   - Each wave loads its 8 Q rows (Step 1)
//   - All 256 threads cooperatively load 1 K tile (32 rows × 128 cols = 4KB)
//   - Output: K[0,0], K[0,127], K[31,0], K[31,127] to verify corners

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16fwd_fp8_256t_s2E
.p2align 8
.type _ZN5aiter16fwd_fp8_256t_s2E,@function

_ZN5aiter16fwd_fp8_256t_s2E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q input [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K input [32×128] FP8
    
    // ========================================================================
    // THREAD DECOMPOSITION (256 threads = 4 waves × 64 lanes)
    // ========================================================================
    v_mov_b32_e32 v10, v0                  // Save original thread_id (0-255)
    v_lshrrev_b32_e32 v1, 6, v0            // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0               // v0 = lane_id (0-63)
    
    // Save wave_id to scalar
    v_readfirstlane_b32 s20, v1            // s20 = wave_id
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // SETUP BUFFER DESCRIPTORS
    // ========================================================================
    s_mov_b32 s10, -1                      // Q size = max
    s_mov_b32 s11, 0x20000                 // Q flags
    
    s_mov_b32 s14, -1                      // K size = max
    s_mov_b32 s15, 0x20000                 // K flags
    
    s_mov_b32 s6, -1                       // O size = max
    s_mov_b32 s7, 0x20000                  // O flags
    
    // ========================================================================
    // STEP 1: WAVE-AWARE Q LOAD (from Step 1)
    // ========================================================================
    // Each wave loads its 8 Q rows (8 × 128 = 1024 bytes) to its LDS region
    v_lshlrev_b32_e32 v2, 4, v0            // v2 = lane_id * 16
    
    s_mul_i32 s21, s20, 1024               // s21 = wave_id * 1024 (LDS base)
    s_mov_b32 m0, s21                      // m0 = LDS destination
    
    buffer_load_dwordx4 v2, s[8:11], s21 offen lds
    
    // ========================================================================
    // STEP 2: SHARED K LOAD (NEW)
    // ========================================================================
    // All 256 threads cooperatively load K tile to LDS[4096:8191]
    // Following 64T pattern: each wave loads 1KB, 4 waves load 4KB total
    //
    // 64T does 4 passes with m0 = 4096, 5120, 6144, 7168
    // 256T does 1 pass with m0 = 4096 + wave_id * 1024
    //
    // Per-thread offset within wave: v3 = lane_id * 16
    // Wave's global source offset: s21 = wave_id * 1024
    // Wave's LDS dest base: m0 = 4096 + wave_id * 1024
    
    // Recalculate per-thread offset (same pattern as 64T)
    v_lshlrev_b32_e32 v3, 4, v0            // v3 = lane_id * 16 (offset within 1KB)
    
    // Calculate wave's K global offset: s21 = wave_id * 1024
    s_mul_i32 s21, s20, 1024               // s21 = wave_id * 1024
    
    // Calculate wave's LDS destination: m0 = 4096 + wave_id * 1024
    s_add_u32 s22, s21, 4096               // s22 = 4096 + wave_id * 1024
    s_mov_b32 m0, s22                      // m0 = LDS base for this wave
    
    // Load K: 
    // - Global source = K_ptr + s21 + v3 = K_ptr + wave_id*1024 + lane_id*16
    // - LDS dest = m0 + v3 = (4096 + wave_id*1024) + lane_id*16
    buffer_load_dwordx4 v3, s[12:15], s21 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // READ BACK AND VERIFY
    // ========================================================================
    // Output for thread 0 only: K corners
    // K[0,0], K[0,127], K[31,0], K[31,127]
    
    // Only thread 0 outputs
    v_cmp_eq_u32_e32 vcc, 0, v10
    s_and_saveexec_b64 s[22:23], vcc
    s_cbranch_execz SKIP_OUTPUT
    
    // Read K[0,0] from LDS[4096]
    v_mov_b32_e32 v5, 4096
    ds_read_u8 v20, v5
    
    // Read K[0,127] from LDS[4096 + 127]
    v_mov_b32_e32 v5, 4096 + 127
    ds_read_u8 v21, v5
    
    // Read K[31,0] from LDS[4096 + 31*128]
    v_mov_b32_e32 v5, 4096 + 31*128
    ds_read_u8 v22, v5
    
    // Read K[31,127] from LDS[4096 + 31*128 + 127]
    v_mov_b32_e32 v5, 4096 + 31*128 + 127
    ds_read_u8 v23, v5
    
    s_waitcnt lgkmcnt(0)
    
    // Convert to F32
    v_cvt_f32_u32_e32 v20, v20
    v_cvt_f32_u32_e32 v21, v21
    v_cvt_f32_u32_e32 v22, v22
    v_cvt_f32_u32_e32 v23, v23
    
    // Store output
    v_mov_b32_e32 v7, 0
    buffer_store_dword v20, v7, s[4:7], 0 offen           // O[0] = K[0,0]
    buffer_store_dword v21, v7, s[4:7], 0 offen offset:4  // O[1] = K[0,127]
    buffer_store_dword v22, v7, s[4:7], 0 offen offset:8  // O[2] = K[31,0]
    buffer_store_dword v23, v7, s[4:7], 0 offen offset:12 // O[3] = K[31,127]
    
SKIP_OUTPUT:
    s_mov_b64 exec, -1
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter16fwd_fp8_256t_s2E, .-_ZN5aiter16fwd_fp8_256t_s2E

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_256t_s2E
    .amdhsa_group_segment_fixed_size 8192
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
  - .name: _ZN5aiter16fwd_fp8_256t_s2E
    .symbol: _ZN5aiter16fwd_fp8_256t_s2E.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 8192
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
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

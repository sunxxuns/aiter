// FP8 Flash Attention - 256 Thread Version (4 Waves)
// O[seq×128] = softmax(Q @ K^T / sqrt(d)) @ V
//
// Architecture:
//   256 threads = 4 waves × 64 threads/wave
//   Each wave processes 8 Q-rows (32 total / 4 waves)
//   All waves share K/V tiles in LDS
//
// Args: O[seq×128], Q[seq×128], K[seq×128], V[seq×128], seq_len
// Launch: grid_x = seq_len / 32, block = 256 threads

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter11fwd_fp8_v256E
.p2align 8
.type _ZN5aiter11fwd_fp8_v256E,@function

_ZN5aiter11fwd_fp8_v256E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // THREAD ID DECOMPOSITION (256 threads = 4 waves)
    // ========================================================================
    // v0 = thread_id (0-255)
    // wave_id = thread_id / 64 (0-3)
    // lane_id = thread_id % 64 (0-63)
    v_lshrrev_b32_e32 v1, 6, v0           // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0              // v0 = lane_id (0-63)
    
    // Each wave handles 8 Q-rows
    // wave 0: rows 0-7,   wave 1: rows 8-15
    // wave 2: rows 16-23, wave 3: rows 24-31
    v_lshlrev_b32_e32 v2, 3, v1           // v2 = wave_id * 8 = row_base
    
    // ========================================================================
    // SAVE WORKGROUP ID (s2 = workgroup_id_x = Q-tile index)
    // ========================================================================
    s_mov_b32 s28, s2                     // Save q_tile_idx
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [seq×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [seq×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [seq×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [seq×128] FP8
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    // Constants
    s_mov_b32 s2, 0x3e028f5c              // log2(e) / sqrt(128) = 0.12754
    s_mov_b32 s3, 0xff800000              // -infinity
    
    s_waitcnt lgkmcnt(0)
    
    // Calculate number of K-tiles
    s_add_i32 s25, s24, 31
    s_lshr_b32 s25, s25, 5                 // num_tiles = (seq_len + 31) / 32
    
    // K/V tile stride
    s_mov_b32 s26, 4096
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE (per-wave)
    // ========================================================================
    v_mov_b32_e32 v70, s3                  // running_max = -inf
    v_mov_b32_e32 v71, 0                   // running_sum = 0
    
    // Initialize O accumulator (v80-v143 for all 4 HD tiles)
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // COMPUTE Q-TILE OFFSET
    // ========================================================================
    s_lshl_b32 s29, s28, 12               // s29 = q_tile_idx * 4096 (Q offset)
    s_lshl_b32 s30, s28, 14               // s30 = q_tile_idx * 16384 (O offset)
    
    s_add_u32 s8, s8, s29
    s_addc_u32 s9, s9, 0
    s_add_u32 s4, s4, s30
    s_addc_u32 s5, s5, 0
    
    // ========================================================================
    // SETUP BUFFER DESCRIPTORS
    // ========================================================================
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000
    
    // ========================================================================
    // LOAD Q TO LDS (all 256 threads cooperate)
    // ========================================================================
    // 256 threads loading 4KB = 16 bytes per thread
    // thread_id (0-255) → offset = thread_id * 16
    v_lshlrev_b32_e32 v3, 4, v0           // lane offset within wave
    v_lshlrev_b32_e32 v4, 10, v1          // wave offset (wave_id * 1024)
    v_add_u32_e32 v3, v3, v4              // total offset
    
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v3, s[8:11], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // K-tile offset and counter
    s_mov_b32 s27, 0                       // k_offset = 0
    s_mov_b32 s31, 0                       // tile_idx = 0

    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
K_TILE_LOOP:
    // Load K tile to LDS at offset 4096 (all 256 threads cooperate)
    v_lshlrev_b32_e32 v3, 4, v0           // lane offset
    v_lshlrev_b32_e32 v4, 10, v1          // wave offset
    v_add_u32_e32 v3, v3, v4
    v_add_u32_e32 v3, 4096, v3            // K at LDS offset 4096
    
    // Load K tile to LDS at offset 4096
    // 256 threads × 16 bytes = 4KB, perfect for one K tile
    // Each thread: LDS[4096 + thread_id*16] = K[k_offset + thread_id*16]
    
    // Compute per-thread offset: thread_id * 16 where thread_id = wave*64 + lane
    v_lshlrev_b32_e32 v5, 6, v1           // wave * 64
    v_add_u32_e32 v5, v0, v5              // thread_id = wave*64 + lane
    v_lshlrev_b32_e32 v5, 4, v5           // thread_id * 16
    
    // LDS offset (VGPR for offen mode)
    v_add_u32_e32 v6, 4096, v5            // 4096 + thread_offset
    
    // Global offset = k_offset + thread_offset
    s_mov_b32 m0, 0
    s_mov_b32 s20, s27                    // base k_offset
    buffer_load_dwordx4 v6, s[12:15], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier

    // ========================================================================
    // QK MFMA: S = K @ Q^T (each wave computes its 8 rows)
    // ========================================================================
    // This wave's Q rows start at: wave_id * 8 * 128 = wave_id * 1024
    // Lane computes: lane_id within 32×32 tile
    
    v_and_b32_e32 v3, 31, v0              // lane % 32
    v_lshrrev_b32_e32 v4, 5, v0           // lane / 32
    
    // Q address in LDS: wave_id * 1024 + lane_id * 16
    v_lshlrev_b32_e32 v5, 10, v1          // wave * 1024
    v_lshlrev_b32_e32 v6, 4, v3           // (lane%32) * 16
    v_add_u32_e32 v5, v5, v6
    
    // Read Q (8 bytes = 8 FP8 values)
    ds_read_b64 v[40:41], v5
    
    // K address in LDS: 4096 + lane * 16
    v_lshlrev_b32_e32 v6, 4, v3
    v_add_u32_e32 v6, 4096, v6
    ds_read_b64 v[42:43], v6
    
    s_waitcnt lgkmcnt(0)
    
    // Pack for MFMA
    v_mov_b32_e32 v44, v40
    v_mov_b32_e32 v45, v41
    
    // Simple QK - just one MFMA for now (will expand)
    v_accvgpr_write_b32 a0, v44
    v_accvgpr_write_b32 a1, v45
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[42:43], 0
    
    s_waitcnt lgkmcnt(0) & vmcnt(0)
    
    // ========================================================================
    // SOFTMAX (simplified for initial version)
    // ========================================================================
    // Find max
    v_max_f32_e32 v20, v32, v33
    v_max_f32_e32 v20, v20, v34
    v_max_f32_e32 v20, v20, v35
    v_max_f32_e32 v20, v20, v36
    v_max_f32_e32 v20, v20, v37
    v_max_f32_e32 v20, v20, v38
    v_max_f32_e32 v20, v20, v39
    v_max_f32_e32 v20, v20, v40
    v_max_f32_e32 v20, v20, v41
    v_max_f32_e32 v20, v20, v42
    v_max_f32_e32 v20, v20, v43
    v_max_f32_e32 v20, v20, v44
    v_max_f32_e32 v20, v20, v45
    v_max_f32_e32 v20, v20, v46
    v_max_f32_e32 v20, v20, v47
    
    // Cross-lane max
    v_max_f32_dpp v20, v20, v20 row_shr:1
    v_max_f32_dpp v20, v20, v20 row_shr:2
    v_max_f32_dpp v20, v20, v20 row_shr:4
    v_max_f32_dpp v20, v20, v20 row_shr:8
    v_max_f32_dpp v20, v20, v20 row_bcast:15
    v_max_f32_dpp v20, v20, v20 row_bcast:31
    
    // tile_max
    v_mov_b32_e32 v21, v20
    
    // Update running_max
    v_max_f32_e32 v22, v70, v21           // new_max = max(running_max, tile_max)
    
    // Correction factor
    v_sub_f32_e32 v23, v70, v22           // old_max - new_max
    v_mul_f32_e32 v23, s2, v23            // * scale
    v_exp_f32_e32 v23, v23                // correction = exp(...)
    
    // Rescale O and running_sum
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    .irp i, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    .irp i, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    .irp i, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
        v_mul_f32_e32 v\i, v23, v\i
    .endr
    v_mul_f32_e32 v71, v23, v71
    
    // Compute P = exp((S - new_max) * scale)
    v_mul_f32_e32 v23, s2, v22            // -new_max * scale
    v_sub_f32_e32 v23, 0, v23             // negate
    
    .irp i, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
        v_mul_f32_e32 v\i, s2, v\i        // S * scale
        v_sub_f32_e32 v\i, v\i, v23       // S*scale - max*scale
        v_exp_f32_e32 v\i, v\i            // P = exp(...)
    .endr
    
    // Sum P
    v_add_f32_e32 v24, v32, v33
    v_add_f32_e32 v24, v24, v34
    v_add_f32_e32 v24, v24, v35
    v_add_f32_e32 v24, v24, v36
    v_add_f32_e32 v24, v24, v37
    v_add_f32_e32 v24, v24, v38
    v_add_f32_e32 v24, v24, v39
    v_add_f32_e32 v24, v24, v40
    v_add_f32_e32 v24, v24, v41
    v_add_f32_e32 v24, v24, v42
    v_add_f32_e32 v24, v24, v43
    v_add_f32_e32 v24, v24, v44
    v_add_f32_e32 v24, v24, v45
    v_add_f32_e32 v24, v24, v46
    v_add_f32_e32 v24, v24, v47
    
    // Cross-lane sum
    v_add_f32_dpp v24, v24, v24 row_shr:1
    v_add_f32_dpp v24, v24, v24 row_shr:2
    v_add_f32_dpp v24, v24, v24 row_shr:4
    v_add_f32_dpp v24, v24, v24 row_shr:8
    v_add_f32_dpp v24, v24, v24 row_bcast:15
    v_add_f32_dpp v24, v24, v24 row_bcast:31
    
    v_add_f32_e32 v71, v71, v24           // running_sum += tile_sum
    v_mov_b32_e32 v70, v22                // running_max = new_max
    
    // Store P to LDS for PV MFMA (simplified)
    // Skip PV for now - just test the structure
    
    // Update K offset for next tile
    s_add_i32 s27, s27, s26               // k_offset += 4096
    s_add_i32 s31, s31, 1                 // tile_idx++
    s_cmp_lt_u32 s31, s25                 // tile_idx < num_tiles?
    s_cbranch_scc1 K_TILE_LOOP

    // ========================================================================
    // FINAL NORMALIZATION
    // ========================================================================
    v_rcp_f32_e32 v71, v71
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mul_f32_e32 v\i, v71, v\i
    .endr
    .irp i, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        v_mul_f32_e32 v\i, v71, v\i
    .endr
    .irp i, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        v_mul_f32_e32 v\i, v71, v\i
    .endr
    .irp i, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143
        v_mul_f32_e32 v\i, v71, v\i
    .endr
    
    // ========================================================================
    // STORE OUTPUT (each wave stores its 8 rows)
    // ========================================================================
    // O descriptor at s[4:7]: base + Q-tile offset already applied
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // Store simplified - just store zeros for now to test structure
    // Real store would use wave_id * 8 * 128 * 4 as offset
    v_lshlrev_b32_e32 v3, 5, v1           // wave_id * 32 (bytes per F32 row piece)
    v_lshlrev_b32_e32 v4, 2, v0           // lane * 4
    v_add_u32_e32 v3, v3, v4
    
    // Store one value per thread (simplified)
    buffer_store_dword v80, v3, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter11fwd_fp8_v256E
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 148
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 148
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
  - .name: _ZN5aiter11fwd_fp8_v256E
    .symbol: _ZN5aiter11fwd_fp8_v256E.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 148
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

// FP8 Flash Attention 256T - Step 1: Structure Validation
// Just load Q and output it to verify thread distribution works
//
// Test: O[i,j] = Q[i,j] (passthrough)

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter16fwd_fp8_v256_s1E
.p2align 8
.type _ZN5aiter16fwd_fp8_v256_s1E,@function

_ZN5aiter16fwd_fp8_v256_s1E:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // THREAD ID: 256 threads = 4 waves × 64 lanes
    // ========================================================================
    v_lshrrev_b32_e32 v1, 6, v0           // v1 = wave_id (0-3)
    v_and_b32_e32 v0, 63, v0              // v0 = lane_id (0-63)
    
    // ========================================================================
    // SAVE WORKGROUP ID
    // ========================================================================
    s_mov_b32 s28, s2                     // q_tile_idx
    
    // ========================================================================
    // LOAD ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K (unused in step1)
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V (unused in step1)
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // COMPUTE OFFSETS
    // ========================================================================
    // Q offset = q_tile_idx * 4096 bytes
    // O offset = q_tile_idx * 16384 bytes (F32 = 4 bytes)
    s_lshl_b32 s29, s28, 12               // Q offset
    s_lshl_b32 s30, s28, 14               // O offset
    
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
    // thread_id = wave*64 + lane (0-255)
    // Each thread loads 16 bytes → 256×16 = 4KB = one Q tile
    
    v_lshlrev_b32_e32 v2, 6, v1           // wave * 64
    v_add_u32_e32 v2, v0, v2              // thread_id
    v_lshlrev_b32_e32 v2, 4, v2           // thread_id * 16
    
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v2, s[8:11], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // TEST: Output thread_id and wave_id pattern
    // ========================================================================
    // Each thread outputs to position based on its ID
    // This validates the thread distribution is correct
    
    // Compute global thread position in output
    // thread_id = wave*64 + lane (0-255)
    // For 32 rows × 128 cols = 4096 elements per tile
    // Map: thread writes to row = thread/128, col = thread%128... 
    // Actually 256 threads can write 256 F32 values per pass
    
    // Simple test: each thread writes its thread_id as F32
    // to position [thread_id / 4, (thread_id % 4) * 32 + (lane % 32)]
    // This way we fill first 8 rows × first 128 cols
    
    // thread_id (0-255)
    v_lshlrev_b32_e32 v3, 6, v1           // wave * 64
    v_add_u32_e32 v3, v0, v3              // thread_id
    
    // Convert thread_id to float for output
    v_cvt_f32_u32_e32 v6, v3              // v6 = (float)thread_id
    
    // Output offset: thread_id * 4 (each F32 is 4 bytes)
    v_lshlrev_b32_e32 v5, 2, v3           // thread_id * 4
    
    // ========================================================================
    // STORE OUTPUT
    // ========================================================================
    buffer_store_dword v6, v5, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter16fwd_fp8_v256_s1E
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
  - .name: _ZN5aiter16fwd_fp8_v256_s1E
    .symbol: _ZN5aiter16fwd_fp8_v256_s1E.kd
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

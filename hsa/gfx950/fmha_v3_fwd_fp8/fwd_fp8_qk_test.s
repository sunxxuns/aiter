// FP8 Flash Attention with K-tile Loop and Online Softmax
// O[32×128] = softmax(Q @ K^T / sqrt(d)) @ V
//
// Online softmax algorithm:
//   running_max = -inf, running_sum = 0, O = 0
//   for each K-tile:
//     S = K_tile @ Q^T (transpose trick)
//     tile_max = max(S)
//     correction = exp((running_max - new_max) * scale)
//     O = O * correction
//     running_sum = running_sum * correction
//     P = exp((S - tile_max) * scale)  // unnormalized
//     tile_sum = sum(P)
//     running_sum += tile_sum
//     running_max = new_max
//     O += P @ V_tile
//   O = O / running_sum
//
// Args: O[32×128], Q[32×128], K[seq×128], V[seq×128], seq_len

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter12fwd_fp8_qktestE
.p2align 8
.type _ZN5aiter12fwd_fp8_qktestE,@function

_ZN5aiter12fwd_fp8_qktestE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [32×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [seq×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [seq×128] FP8
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    v_and_b32_e32 v0, 63, v0
    
    // Constants
    s_mov_b32 s2, 0x3e028f5c              // log2(e) / sqrt(128) = 0.12754
    s_mov_b32 s3, 0xff800000              // -infinity
    
    s_waitcnt lgkmcnt(0)
    
    // Calculate number of K-tiles
    s_add_i32 s25, s24, 31
    s_lshr_b32 s25, s25, 5                 // num_tiles = (seq_len + 31) / 32
    
    // K/V tile stride (32 rows × 128 cols × 1 byte = 4096 bytes)
    s_mov_b32 s26, 4096
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE
    // ========================================================================
    v_mov_b32_e32 v70, s3                  // running_max = -inf
    v_mov_b32_e32 v71, 0                   // running_sum = 0
    
    // Initialize O accumulator (v80-v143 for all 4 HD tiles)
    // HD tile 0 (cols 0-31):   v[80:95]
    // HD tile 1 (cols 32-63):  v[96:111]
    // HD tile 2 (cols 64-95):  v[112:127]
    // HD tile 3 (cols 96-127): v[128:143]
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
    // SETUP BUFFER DESCRIPTORS (4 SGPRs each: base_lo, base_hi, size, flags)
    // ========================================================================
    // Q descriptor at s[8:11]: s[8:9] already has Q pointer
    s_mov_b32 s10, -1              // size = max
    s_mov_b32 s11, 0x20000         // flags
    
    // K descriptor at s[12:15]: s[12:13] already has K pointer
    s_mov_b32 s14, -1              // size = max
    s_mov_b32 s15, 0x20000         // flags
    
    // V descriptor at s[16:19]: s[16:17] already has V pointer
    s_mov_b32 s18, -1              // size = max
    s_mov_b32 s19, 0x20000         // flags
    
    // ========================================================================
    // LOAD Q TO LDS (stays for all tiles)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v0
    
    s_mov_b32 m0, 0
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 1024
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 2048
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    s_mov_b32 m0, 3072
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // K-tile offset and counter
    s_mov_b32 s27, 0                       // k_offset = 0
    s_mov_b32 s28, 0                       // tile_idx = 0
    
    // K and V buffer descriptors already set up above with size=-1 (max)
    // DO NOT overwrite s14, s18 - they need to allow access beyond 4096 bytes
    // for multi-tile access with s27 offset

    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
K_TILE_LOOP:
    // Recalculate v1 for K load (might have been clobbered)
    v_lshlrev_b32_e32 v1, 4, v0
    
    // Load K tile to LDS at offset 4096 (matching working kernel pattern)
    s_mov_b32 m0, 4096
    s_mov_b32 s20, s27
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 5120
    s_add_i32 s20, s27, 1024
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 6144
    s_add_i32 s20, s27, 2048
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    s_mov_b32 m0, 7168
    s_add_i32 s20, s27, 3072
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // QK MFMA: S^T[32×32] = K @ Q^T
    // ========================================================================
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
    
    // Read Q and K from LDS (same pattern as working kernel)
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4
    v_add_u32_e32 v6, 4096, v5            // K base at LDS offset 4096
    
    // 8 QK MFMA passes: S^T = K @ Q^T
    // A operand = K, B operand = Q (swapped for transpose)
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v7, \k_off, v6      // K offset (A operand)
        v_add_u32_e32 v8, \k_off, v5      // Q offset (B operand)
        ds_read_b64 v[20:21], v7          // Read K for A operand
        ds_read_b64 v[22:23], v8          // Read Q for B operand
        s_waitcnt lgkmcnt(0)
        v_accvgpr_write_b32 a0, v20       // K → A
        v_accvgpr_write_b32 a1, v21
        s_nop 1
        v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
        s_nop 15
    .endr
    s_nop 15
    
    // ========================================================================
    // ONLINE SOFTMAX UPDATE
    // ========================================================================
    
    // Find tile_max
    v_max_f32_e32 v21, v32, v33
    v_max_f32_e32 v21, v21, v34
    v_max_f32_e32 v21, v21, v35
    v_max_f32_e32 v21, v21, v36
    v_max_f32_e32 v21, v21, v37
    // ========================================================================
    // OUTPUT S (instead of continuing with softmax)
    // ========================================================================
    // Output raw S values v[32:35] from lane 0
    v_cmp_eq_u32_e32 vcc, 0, v0
    s_and_saveexec_b64 s[28:29], vcc
    s_cbranch_execz END_KERNEL_QK
    
    v_mov_b32_e32 v1, 0
    buffer_store_dword v32, v1, s[4:7], 0 offen offset:0
    buffer_store_dword v33, v1, s[4:7], 0 offen offset:4
    buffer_store_dword v34, v1, s[4:7], 0 offen offset:8
    buffer_store_dword v35, v1, s[4:7], 0 offen offset:12
    
END_KERNEL_QK:
    s_mov_b64 exec, -1
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter12fwd_fp8_qktestE, .-_ZN5aiter12fwd_fp8_qktestE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter12fwd_fp8_qktestE
    .amdhsa_group_segment_fixed_size 12288
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
  - .name: _ZN5aiter12fwd_fp8_qktestE
    .symbol: _ZN5aiter12fwd_fp8_qktestE.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 148
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

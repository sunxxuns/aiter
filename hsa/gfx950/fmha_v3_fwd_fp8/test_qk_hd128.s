// QK MFMA test for HD=128: S[32×32] = Q[32×128] @ K^T[128×32]
// Uses buffer_load→LDS pattern
// 8 MFMA passes accumulating into same S output

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter12test_qk_hd128E
.p2align 8
.type _ZN5aiter12test_qk_hd128E,@function

_ZN5aiter12test_qk_hd128E:
    s_mov_b64 exec, -1
    
    // Load kernel args
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // S output [32×32] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [32×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [32×128] FP8
    
    v_and_b32_e32 v0, 63, v0
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // Load Q[32×128] to LDS at offset 0 (4KB)
    // 64 threads × 16 bytes per load × 4 loads = 4096 bytes
    // Use soffset (SGPR) to advance global address, m0 to advance LDS address
    // ========================================================================
    s_mov_b32 s10, 4096
    s_mov_b32 s11, 0x20000
    s_mov_b32 s20, 0           // soffset for global address
    
    v_lshlrev_b32_e32 v1, 4, v0    // voffset = tid * 16
    
    // Load 0: global[0:1024] → LDS[0:1024]
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    // Load 1: global[1024:2048] → LDS[1024:2048]
    s_mov_b32 m0, 1024
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    // Load 2: global[2048:3072] → LDS[2048:3072]
    s_mov_b32 m0, 2048
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    // Load 3: global[3072:4096] → LDS[3072:4096]
    s_mov_b32 m0, 3072
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[8:11], s20 offen lds
    
    // ========================================================================
    // Load K[32×128] to LDS at offset 4096 (4KB)
    // ========================================================================
    s_mov_b32 s14, 4096
    s_mov_b32 s15, 0x20000
    
    // Load 0: global[0:1024] → LDS[4096:5120]
    s_mov_b32 m0, 4096
    s_mov_b32 s20, 0
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    // Load 1: global[1024:2048] → LDS[5120:6144]
    s_mov_b32 m0, 5120
    s_mov_b32 s20, 1024
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    // Load 2: global[2048:3072] → LDS[6144:7168]
    s_mov_b32 m0, 6144
    s_mov_b32 s20, 2048
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    // Load 3: global[3072:4096] → LDS[7168:8192]
    s_mov_b32 m0, 7168
    s_mov_b32 s20, 3072
    buffer_load_dwordx4 v1, s[12:15], s20 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // ========================================================================
    // QK MFMA - 8 passes
    // Q layout in LDS: row-major [32×128], Q[r,c] at LDS[r*128 + c]
    // K layout in LDS: row-major [32×128], K[r,c] at LDS[4096 + r*128 + c]
    // MFMA 32×32×16 FP8: needs 8 FP8 (8 bytes) per thread for A and B inputs
    // 
    // For S = Q @ K^T:
    //   S[i,j] = sum_k Q[i,k] * K[j,k]
    // So for MFMA:
    //   A input: Q values along K dimension
    //   B input: K values along K dimension (same k indices as A)
    // ========================================================================
    
    // Initialize S accumulators
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
    
    // Thread → row mapping for MFMA
    // From HD=32 working code analysis:
    //   Thread tid reads from row (tid % 32) of Q and K
    //   Within each row, threads 0-31 read k offsets 0..7
    //   and threads 32-63 read k offsets 8..15 for each MFMA pass
    v_and_b32_e32 v2, 31, v0              // row = tid % 32
    v_lshrrev_b32_e32 v3, 5, v0           // half = tid / 32 (0 or 1)
    
    // Q base offset for thread:
    //   row * 128 (row stride for HD=128) + half * 8 (k offset within 16)
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4              // Q base for this thread
    
    // K base offset: same row/half mapping, starting at LDS offset 4096
    v_add_u32_e32 v6, 4096, v5
    
    // ========== MFMA Pass 0: k=0..15 ==========
    // Each thread reads 8 FP8 from Q and 8 FP8 from K
    // v5 base already accounts for half*8, so pass 0 reads k=[0..7] or k=[8..15]
    ds_read_b64 v[20:21], v5
    ds_read_b64 v[22:23], v6
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 1: k=16..31 ==========
    // Advance by 16 bytes (one MFMA K-chunk)
    v_add_u32_e32 v7, 16, v5
    v_add_u32_e32 v8, 16, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 2: k=32..47 ==========
    v_add_u32_e32 v7, 32, v5
    v_add_u32_e32 v8, 32, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 3: k=48..63 ==========
    v_add_u32_e32 v7, 48, v5
    v_add_u32_e32 v8, 48, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 4: k=64..79 ==========
    v_add_u32_e32 v7, 64, v5
    v_add_u32_e32 v8, 64, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 5: k=80..95 ==========
    v_add_u32_e32 v7, 80, v5
    v_add_u32_e32 v8, 80, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 6: k=96..111 ==========
    v_add_u32_e32 v7, 96, v5
    v_add_u32_e32 v8, 96, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // ========== MFMA Pass 7: k=112..127 ==========
    v_add_u32_e32 v7, 112, v5
    v_add_u32_e32 v8, 112, v6
    ds_read_b64 v[20:21], v7
    ds_read_b64 v[22:23], v8
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    s_nop 15
    
    // ========================================================================
    // Store S[32×32] with correct MFMA output layout
    // row = ((vreg-32) % 4) + (tid//32)*4 + ((vreg-32)//4)*8
    // col = tid % 32
    // ========================================================================
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v3, 2, v3
    
    .macro STORE_S vreg, row_mod4, row_8_group
        v_mov_b32_e32 v7, \row_mod4
        v_add_u32_e32 v7, v7, v3
        v_add_u32_e32 v7, \row_8_group * 8, v7
        v_lshlrev_b32_e32 v7, 5, v7
        v_add_u32_e32 v7, v7, v2
        v_lshlrev_b32_e32 v7, 2, v7
        v_mov_b32_e32 v10, s4
        v_mov_b32_e32 v11, s5
        v_add_co_u32_e32 v10, vcc, v7, v10
        v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
        flat_store_dword v[10:11], \vreg
    .endm
    
    STORE_S v32, 0, 0
    STORE_S v33, 1, 0
    STORE_S v34, 2, 0
    STORE_S v35, 3, 0
    STORE_S v36, 0, 1
    STORE_S v37, 1, 1
    STORE_S v38, 2, 1
    STORE_S v39, 3, 1
    STORE_S v40, 0, 2
    STORE_S v41, 1, 2
    STORE_S v42, 2, 2
    STORE_S v43, 3, 2
    STORE_S v44, 0, 3
    STORE_S v45, 1, 3
    STORE_S v46, 2, 3
    STORE_S v47, 3, 3
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter12test_qk_hd128E
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 52
    .amdhsa_next_free_sgpr 24
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
  - .name: _ZN5aiter12test_qk_hd128E
    .symbol: _ZN5aiter12test_qk_hd128E.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 52
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .args:
      - {.name: ptr_S, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

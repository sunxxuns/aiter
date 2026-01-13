// FP8 QK MFMA with 256 threads (4 waves) - matching BF16 pattern
// Each wave handles one quadrant of the 32x32 output

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set LDS_STRIDE, 128
.set HD, 128
.set Q_LDS_BASE, 0
.set K_LDS_BASE, 4096

.text
.globl _ZN5aiter11qk_fp8_256tE
.p2align 8
.type _ZN5aiter11qk_fp8_256tE,@function

_ZN5aiter11qk_fp8_256tE:
    s_mov_b64 exec, -1
    
    s_load_dwordx4 s[4:7], s[0:1], 0     // output, K_ptr
    s_load_dwordx2 s[10:11], s[0:1], 16  // Q_ptr
    s_waitcnt lgkmcnt(0)
    
    // tid = thread ID within block (0-255)
    // wave_id = tid / 64 (0-3)
    // lane_id = tid % 64 (0-63)
    v_mov_b32_e32 v60, v0                // tid
    v_lshrrev_b32_e32 v61, 6, v0         // wave_id = tid >> 6
    v_and_b32_e32 v62, 63, v0            // lane_id = tid & 63
    
    // ========================================================================
    // COOPERATIVE LOAD: All 256 threads load Q and K
    // Each thread loads 128/8 = 16 bytes (one row slice)
    // Wave 0-3 each handle 8 rows, covering all 32 rows
    // ========================================================================
    
    // Each thread handles: row = (wave_id * 8) + (lane_id / 8)
    //                      col_start = (lane_id % 8) * 16
    v_lshrrev_b32_e32 v1, 3, v62         // lane_id / 8 (0-7)
    v_lshlrev_b32_e32 v2, 3, v61         // wave_id * 8
    v_add_u32_e32 v3, v1, v2             // row (0-31)
    
    v_and_b32_e32 v4, 7, v62             // lane_id % 8 (0-7)
    v_lshlrev_b32_e32 v4, 4, v4          // col_start = (lane_id % 8) * 16
    
    // Global Q address: Q_ptr + row * 128 + col_start
    v_lshlrev_b32_e32 v5, 7, v3          // row * 128
    v_add_u32_e32 v5, v5, v4             // + col_start
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v5, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS Q address: row * 128 + col_start
    v_lshlrev_b32_e32 v6, 7, v3
    v_add_u32_e32 v6, v6, v4
    
    // Load Q (16 bytes per thread)
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Global K address: K_ptr + row * 128 + col_start
    v_lshlrev_b32_e32 v5, 7, v3
    v_add_u32_e32 v5, v5, v4
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v5, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS K address: K_LDS_BASE + row * 128 + col_start
    v_lshlrev_b32_e32 v6, 7, v3
    v_add_u32_e32 v6, v6, v4
    v_add_u32_e32 v6, K_LDS_BASE, v6
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE: Each wave computes full 32x32 MFMA (same as 64T version)
    // All 4 waves do identical computation (redundant but matches BF16 pattern)
    // ========================================================================
    
    // MFMA row mapping using lane_id (not tid)
    v_and_b32_e32 v1, 3, v62
    v_lshrrev_b32_e32 v2, 3, v62
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    
    v_lshrrev_b32_e32 v4, 2, v62
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v63, v3, v4            // full_row
    
    v_mov_b32_e32 v64, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v62, 32
    v_cndmask_b32_e32 v64, v64, v4, vcc  // k_base
    
    // LDS addresses
    v_lshlrev_b32_e32 v65, 7, v63        // row * 128
    v_add_u32_e32 v65, v65, v64          // + k_base
    
    v_add_u32_e32 v70, K_LDS_BASE, v65   // K addr
    v_mov_b32_e32 v71, v65               // Q addr
    
    // Clear accumulators
    v_mov_b32_e32 v0, 0
    v_mov_b32_e32 v1, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v6, 0
    v_mov_b32_e32 v7, 0
    v_mov_b32_e32 v8, 0
    v_mov_b32_e32 v9, 0
    v_mov_b32_e32 v10, 0
    v_mov_b32_e32 v11, 0
    v_mov_b32_e32 v12, 0
    v_mov_b32_e32 v13, 0
    v_mov_b32_e32 v14, 0
    v_mov_b32_e32 v15, 0
    
    // 8 MFMAs for HD=128
    .irp k_off, 0, 16, 32, 48, 64, 80, 96, 112
        v_add_u32_e32 v72, \k_off, v70
        ds_read_b64 v[30:31], v72
        v_add_u32_e32 v73, \k_off, v71
        ds_read_b64 v[32:33], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    
    s_nop 15
    
    // ========================================================================
    // OUTPUT: Only wave 0 writes results
    // ========================================================================
    
    v_cmp_eq_u32_e64 vcc, v61, 0         // wave_id == 0?
    s_and_saveexec_b64 s[12:13], vcc
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v62        // lane_id * 64
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx4 v[40:41], v[0:3]
    v_add_co_u32_e32 v42, vcc, 16, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[4:7]
    v_add_co_u32_e32 v42, vcc, 32, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[8:11]
    v_add_co_u32_e32 v42, vcc, 48, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[12:15]
    
    s_mov_b64 exec, s[12:13]
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter11qk_fp8_256tE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter11qk_fp8_256tE
    .symbol: _ZN5aiter11qk_fp8_256tE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 80
    .max_flat_workgroup_size: 256
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

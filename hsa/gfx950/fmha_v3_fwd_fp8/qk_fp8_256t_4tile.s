// FP8 QK MFMA with 256 threads - 4 output tiles (proper 4x work)
// Each wave computes a different 32x32 tile: S[wave_id*32:(wave_id+1)*32, :]
// Total: 128x32 output from 128x128 K × 32x128 Q^T

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set LDS_STRIDE, 128
.set HD, 128
.set Q_LDS_BASE, 0           // Q: 32×128 = 4096 bytes
.set K_LDS_BASE, 4096        // K: 128×128 = 16384 bytes (4 tiles of 32×128)

.text
.globl _ZN5aiter17qk_fp8_256t_4tileE
.p2align 8
.type _ZN5aiter17qk_fp8_256t_4tileE,@function

_ZN5aiter17qk_fp8_256t_4tileE:
    s_mov_b64 exec, -1
    
    // Args: output (128×32 floats), K_ptr (128×128 fp8), Q_ptr (32×128 fp8)
    s_load_dwordx4 s[4:7], s[0:1], 0
    s_load_dwordx2 s[10:11], s[0:1], 16
    s_waitcnt lgkmcnt(0)
    
    v_mov_b32_e32 v60, v0                // tid (0-255)
    v_lshrrev_b32_e32 v61, 6, v0         // wave_id (0-3)
    v_and_b32_e32 v62, 63, v0            // lane_id (0-63)
    
    // ========================================================================
    // LOAD Q[32×128] - same Q for all waves
    // Each of 256 threads loads 4096/256 = 16 bytes
    // ========================================================================
    
    // Thread tid loads bytes [tid*16 : tid*16+16)
    v_lshlrev_b32_e32 v1, 4, v60         // tid * 16
    
    // Global addr
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, s11
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K[128×128] - 4 tiles, each wave loads its own 32×128 tile
    // Wave w loads K[w*32:(w+1)*32, :]
    // ========================================================================
    
    // Each wave: 32×128 = 4096 bytes, 64 threads, each loads 64 bytes (4 loads)
    // Global K offset: wave_id * 32 * 128 = wave_id * 4096
    v_lshlrev_b32_e32 v1, 12, v61        // wave_id * 4096
    
    // Within wave: lane loads row (lane/2), cols (lane%2)*64
    v_lshrrev_b32_e32 v2, 1, v62         // lane / 2 = row within tile (0-31)
    v_and_b32_e32 v3, 1, v62             // lane % 2 = col half (0 or 1)
    v_lshlrev_b32_e32 v3, 6, v3          // col_start = 0 or 64
    
    // Global addr: K_ptr + wave_id*4096 + row*128 + col_start
    v_lshlrev_b32_e32 v4, 7, v2          // row * 128
    v_add_u32_e32 v4, v4, v3             // + col_start
    v_add_u32_e32 v4, v4, v1             // + wave_offset
    
    v_mov_b32_e32 v10, s6
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS addr: K_LDS_BASE + wave_id*4096 + row*128 + col_start
    v_add_u32_e32 v5, K_LDS_BASE, v4
    
    // Load 64 bytes in 4 loads
    flat_load_dwordx4 v[20:23], v[10:11] offset:0
    flat_load_dwordx4 v[24:27], v[10:11] offset:16
    flat_load_dwordx4 v[28:31], v[10:11] offset:32
    flat_load_dwordx4 v[32:35], v[10:11] offset:48
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v6, 0, v5
    ds_write_b128 v6, v[20:23]
    v_add_u32_e32 v6, 16, v5
    ds_write_b128 v6, v[24:27]
    v_add_u32_e32 v6, 32, v5
    ds_write_b128 v6, v[28:31]
    v_add_u32_e32 v6, 48, v5
    ds_write_b128 v6, v[32:35]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE: Each wave computes its own 32×32 tile
    // Wave w: S[w*32:(w+1)*32, :] = K[w*32:(w+1)*32, :] @ Q^T
    // ========================================================================
    
    // MFMA row mapping (same for all waves)
    v_and_b32_e32 v1, 3, v62
    v_lshrrev_b32_e32 v2, 3, v62
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    
    v_lshrrev_b32_e32 v4, 2, v62
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v63, v3, v4            // mfma_row (0-31)
    
    v_mov_b32_e32 v64, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v62, 32
    v_cndmask_b32_e32 v64, v64, v4, vcc  // k_base
    
    // K LDS addr: K_LDS_BASE + wave_id*4096 + mfma_row*128 + k_base
    v_lshlrev_b32_e32 v65, 12, v61       // wave_id * 4096
    v_add_u32_e32 v65, K_LDS_BASE, v65
    v_lshlrev_b32_e32 v66, 7, v63        // mfma_row * 128
    v_add_u32_e32 v65, v65, v66
    v_add_u32_e32 v70, v65, v64          // + k_base
    
    // Q LDS addr: Q_LDS_BASE + mfma_row*128 + k_base (same Q for all waves)
    v_lshlrev_b32_e32 v66, 7, v63
    v_add_u32_e32 v71, v66, v64
    
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
    // OUTPUT: Each wave writes its 32×32 tile
    // Wave w writes to output[w*32*32 : (w+1)*32*32] (each lane writes 16 floats)
    // ========================================================================
    
    // Output offset: wave_id * 32 * 32 * 4 = wave_id * 4096 bytes
    //              + lane_id * 64 bytes
    v_lshlrev_b32_e32 v40, 12, v61       // wave_id * 4096
    v_lshlrev_b32_e32 v41, 6, v62        // lane_id * 64
    v_add_u32_e32 v40, v40, v41
    
    v_mov_b32_e32 v42, s4
    v_mov_b32_e32 v43, s5
    v_add_co_u32_e32 v42, vcc, v40, v42
    v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
    
    flat_store_dwordx4 v[42:43], v[0:3]
    v_add_co_u32_e32 v44, vcc, 16, v42
    v_addc_co_u32_e32 v45, vcc, 0, v43, vcc
    flat_store_dwordx4 v[44:45], v[4:7]
    v_add_co_u32_e32 v44, vcc, 32, v42
    v_addc_co_u32_e32 v45, vcc, 0, v43, vcc
    flat_store_dwordx4 v[44:45], v[8:11]
    v_add_co_u32_e32 v44, vcc, 48, v42
    v_addc_co_u32_e32 v45, vcc, 0, v43, vcc
    flat_store_dwordx4 v[44:45], v[12:15]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter17qk_fp8_256t_4tileE
    .amdhsa_group_segment_fixed_size 24576
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
  - .name: _ZN5aiter17qk_fp8_256t_4tileE
    .symbol: _ZN5aiter17qk_fp8_256t_4tileE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 24576
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

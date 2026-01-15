// FP8 QK MFMA with row-major LDS (Step 1: verify correctness before swizzle)
// Computes S = Q @ K^T for a 32x32 tile
// 256 threads (4 waves), each thread contributes to MFMA

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter15fwd_fp8_swizzleE
.p2align 8
.type _ZN5aiter15fwd_fp8_swizzleE,@function

_ZN5aiter15fwd_fp8_swizzleE:
    s_mov_b64 exec, -1
    
    // Args: output, Q_ptr, K_ptr
    s_load_dwordx2 s[4:5], s[0:1], 0      // output
    s_load_dwordx2 s[8:9], s[0:1], 8      // Q_ptr
    s_load_dwordx2 s[12:13], s[0:1], 16   // K_ptr
    s_waitcnt lgkmcnt(0)
    
    // Thread info - only use lane within wave
    v_and_b32_e32 v60, 63, v0             // lane_id (0-63)
    
    // ========================================================================
    // SETUP BUFFER DESCRIPTORS
    // ========================================================================
    // Q descriptor: s[8:11]
    s_mov_b32 s10, -1                     // size = unlimited
    s_mov_b32 s11, 0x20000                // OOB returns 0
    
    // K descriptor: s[12:15]
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    
    // ========================================================================
    // LOAD Q[32×128] TO LDS (64 threads load 4096 bytes)
    // Each thread loads 16 bytes at offset tid * 16
    // LDS destination controlled by m0
    // ========================================================================
    
    // Per-thread offset: lane * 16
    v_lshlrev_b32_e32 v1, 4, v60          // lane * 16
    
    // Load Q to LDS using buffer_load ... lds
    // m0 = LDS destination base, s20 = global offset base
    // 64 threads load 64 * 16 = 1024 bytes per wave
    // 4 waves = 4096 bytes = Q[32×128]
    
    // Wave 0: LDS[0..1023], Wave 1: LDS[1024..2047], etc.
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id = tid / 64
    v_lshlrev_b32_e32 v2, 10, v61         // wave_id * 1024
    v_add_u32_e32 v3, v1, v2              // LDS offset = lane*16 + wave*1024
    
    // Set m0 for LDS destination (must be scalar for buffer_load...lds)
    // Use wave-uniform value
    v_readfirstlane_b32 s20, v3           // Can't do this - each lane needs different m0
    
    // ALTERNATIVE: Use flat_load + ds_write pattern (per-lane addressing)
    // Global Q addr
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_u32_e32 v4, v1, v2              // total offset = lane*16 + wave*1024
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS Q addr = same offset (Q at LDS[0..4095])
    v_mov_b32_e32 v5, v4
    
    // Load from global, write to LDS
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v5, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K[32×128] TO LDS (at offset 4096)
    // ========================================================================
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v4, v10    // Same offset pattern
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // LDS K addr = 4096 + offset
    v_add_u32_e32 v6, 4096, v4
    
    flat_load_dwordx4 v[24:27], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v6, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE MFMA ROW MAPPING (for each lane in wave)
    // For v_mfma_f32_32x32x16_fp8_fp8:
    // row = (lane & 3) + ((lane >> 3) & 3) * 4 + ((lane >> 2) & 1) * 16
    // k_half = 0 for lanes 0-31, 8 for lanes 32-63
    // ========================================================================
    
    v_and_b32_e32 v1, 3, v60              // lane & 3
    v_lshrrev_b32_e32 v2, 3, v60          // lane >> 3
    v_and_b32_e32 v2, 3, v2               // & 3
    v_lshlrev_b32_e32 v2, 2, v2           // * 4
    v_add_u32_e32 v3, v1, v2              // row16
    
    v_lshrrev_b32_e32 v4, 2, v60          // lane >> 2
    v_and_b32_e32 v4, 1, v4               // & 1
    v_lshlrev_b32_e32 v4, 4, v4           // * 16
    v_add_u32_e32 v63, v3, v4             // v63 = mfma_row (0-31)
    
    // k_half: 0 for lanes 0-31, 8 for lanes 32-63
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc
    
    // LDS read addresses
    // Q: mfma_row * 128 + k_half (Q at LDS[0..4095])
    // K: 4096 + mfma_row * 128 + k_half
    v_lshlrev_b32_e32 v10, 7, v63         // mfma_row * 128
    v_add_u32_e32 v10, v10, v64           // + k_half
    
    v_mov_b32_e32 v70, v10                // Q addr base
    v_add_u32_e32 v71, 4096, v10          // K addr base
    
    // ========================================================================
    // CLEAR ACCUMULATORS
    // ========================================================================
    
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // ========================================================================
    // 8 MFMA ITERATIONS FOR HD=128
    // ========================================================================
    
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
    // STORE OUTPUT
    // Each of 64 lanes stores 16 F32 values (64 bytes)
    // Output layout: lane i stores at output + lane * 64
    // ========================================================================
    
    v_mov_b32_e32 v40, s4
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60         // lane * 64
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
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter15fwd_fp8_swizzleE
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 80
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter15fwd_fp8_swizzleE
    .symbol: _ZN5aiter15fwd_fp8_swizzleE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 80
    .max_flat_workgroup_size: 256
    .args:
      - {.name: output, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

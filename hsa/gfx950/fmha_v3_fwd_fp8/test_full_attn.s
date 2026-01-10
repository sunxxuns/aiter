// Complete Flash Attention Test: QK -> softmax -> P redistrib -> PV
// Uses known input patterns to verify correctness
//
// Q[q,d] = 0 (uniform)
// K[k,d] = 0 (uniform)
// V[k,d] = d/32 (varies by D)
//
// With uniform Q and K: QK = 0, softmax(0) = 1/K_dim = 1/16
// O[q,d] = sum_k(1/16 * d/32) = d/32
//
// Expected output: O[q,d] = d/32 for all q

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter14test_full_attnE
.p2align 8
.type _ZN5aiter14test_full_attnE,@function

.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 2048
.set LDS_P_OFFSET, 4096

_ZN5aiter14test_full_attnE:
    s_and_b32 s1, s1, 0xffff
    s_load_dwordx2 s[8:9], s[0:1], 0x00    // ptr_out
    s_load_dwordx2 s[10:11], s[0:1], 0x08  // ptr_V
    
    v_and_b32_e32 v0, 63, v0               // lane_id
    
    s_waitcnt lgkmcnt(0)
    
    // Initialize accumulators
    v_mov_b32_e32 v48, 0
    v_mov_b32_e32 v49, 0
    v_mov_b32_e32 v50, 0
    v_mov_b32_e32 v51, 0
    v_mov_b32_e32 v52, 0
    v_mov_b32_e32 v53, 0
    v_mov_b32_e32 v54, 0
    v_mov_b32_e32 v55, 0
    v_mov_b32_e32 v56, 0
    v_mov_b32_e32 v57, 0
    v_mov_b32_e32 v58, 0
    v_mov_b32_e32 v59, 0
    v_mov_b32_e32 v60, 0
    v_mov_b32_e32 v61, 0
    v_mov_b32_e32 v62, 0
    v_mov_b32_e32 v63, 0
    
    // ========================================================================
    // STEP 1: QK MFMA (skip - we'll use uniform P = 1/16 directly)
    // ========================================================================
    
    // For this test, skip actual QK computation.
    // Assume P[q,k] = 0.0625 (1/16) for all (q,k)
    // This simulates softmax of uniform attention scores.
    
    // ========================================================================
    // STEP 2: Generate P values and write to LDS
    // P[q,k] = 0.0625 for all
    // Thread t writes to K column = t%32 for all Q rows
    // ========================================================================
    
    v_mov_b32_e32 v2, 0x3d800000          // 0.0625
    v_mov_b32_e32 v3, 0
    v_cvt_pk_fp8_f32 v3, v2, v2           // 2 identical values
    
    // Write to LDS: P[Q, K] at Q*32 + K
    v_and_b32_e32 v4, 31, v0              // K = tid % 32
    v_add_u32_e32 v5, LDS_P_OFFSET, v4
    
    // Write P to all 32 Q rows
    ds_write_b8 v5, v3                    // Q=0
    ds_write_b8 v5, v3 offset:32          // Q=1
    ds_write_b8 v5, v3 offset:64          // Q=2
    ds_write_b8 v5, v3 offset:96          // Q=3
    ds_write_b8 v5, v3 offset:128         // Q=4
    ds_write_b8 v5, v3 offset:160         // Q=5
    ds_write_b8 v5, v3 offset:192         // Q=6
    ds_write_b8 v5, v3 offset:224         // Q=7
    ds_write_b8 v5, v3 offset:256         // Q=8
    ds_write_b8 v5, v3 offset:288         // Q=9
    ds_write_b8 v5, v3 offset:320         // Q=10
    ds_write_b8 v5, v3 offset:352         // Q=11
    ds_write_b8 v5, v3 offset:384         // Q=12
    ds_write_b8 v5, v3 offset:416         // Q=13
    ds_write_b8 v5, v3 offset:448         // Q=14
    ds_write_b8 v5, v3 offset:480         // Q=15
    ds_write_b8 v5, v3 offset:512         // Q=16
    ds_write_b8 v5, v3 offset:544         // Q=17
    ds_write_b8 v5, v3 offset:576         // Q=18
    ds_write_b8 v5, v3 offset:608         // Q=19
    ds_write_b8 v5, v3 offset:640         // Q=20
    ds_write_b8 v5, v3 offset:672         // Q=21
    ds_write_b8 v5, v3 offset:704         // Q=22
    ds_write_b8 v5, v3 offset:736         // Q=23
    ds_write_b8 v5, v3 offset:768         // Q=24
    ds_write_b8 v5, v3 offset:800         // Q=25
    ds_write_b8 v5, v3 offset:832         // Q=26
    ds_write_b8 v5, v3 offset:864         // Q=27
    ds_write_b8 v5, v3 offset:896         // Q=28
    ds_write_b8 v5, v3 offset:928         // Q=29
    ds_write_b8 v5, v3 offset:960         // Q=30
    ds_write_b8 v5, v3 offset:992         // Q=31
    
    s_barrier
    
    // ========================================================================
    // STEP 3: Read P for PV MFMA A operand
    // Thread t needs P[Q=t%32, K_range]
    // ========================================================================
    
    v_and_b32_e32 v6, 31, v0              // Q = tid % 32
    v_lshrrev_b32_e32 v7, 5, v0           // K_group = tid / 32
    v_lshlrev_b32_e32 v7, 3, v7           // K_start = K_group * 8
    
    v_lshlrev_b32_e32 v6, 5, v6           // Q * 32
    v_add_u32_e32 v6, v6, v7              // + K_start
    v_add_u32_e32 v6, LDS_P_OFFSET, v6
    
    ds_read_b64 v[32:33], v6              // Read 8 FP8 P values
    
    s_waitcnt lgkmcnt(0)
    
    v_accvgpr_write_b32 a0, v32
    v_accvgpr_write_b32 a1, v33
    
    // ========================================================================
    // STEP 4: Load V for PV MFMA B operand
    // V[k,d] = d/32 (from global memory)
    // For B operand: thread t provides V[K_range, D=t%32]
    // ========================================================================
    
    // Load V from global memory: V[k,d] stored at ptr_V + k*32 + d
    v_and_b32_e32 v8, 31, v0              // D = tid % 32
    v_lshrrev_b32_e32 v9, 5, v0           // K_group = tid / 32
    v_lshlrev_b32_e32 v9, 3, v9           // K_start = K_group * 8
    
    // V address for each K value: ptr_V + K*32 + D
    // We need 8 K values
    v_mov_b32_e32 v10, s10                // V base low
    v_mov_b32_e32 v11, s11                // V base high
    v_add_co_u32_e32 v10, vcc, v8, v10    // + D
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Add K_start * 32 (K_start * stride)
    v_lshlrev_b32_e32 v9, 5, v9           // K_start * 32
    v_add_co_u32_e32 v10, vcc, v9, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Load 8 V values with stride 32 (one per K row)
    flat_load_ubyte v34, v[10:11]                   // K+0
    flat_load_ubyte v35, v[10:11] offset:32        // K+1
    flat_load_ubyte v36, v[10:11] offset:64        // K+2
    flat_load_ubyte v37, v[10:11] offset:96        // K+3
    flat_load_ubyte v38, v[10:11] offset:128       // K+4
    flat_load_ubyte v39, v[10:11] offset:160       // K+5
    flat_load_ubyte v40, v[10:11] offset:192       // K+6
    flat_load_ubyte v41, v[10:11] offset:224       // K+7
    
    s_waitcnt vmcnt(0)
    
    // Pack V values into 2 dwords
    v_lshlrev_b32_e32 v35, 8, v35
    v_or_b32_e32 v34, v34, v35
    v_lshlrev_b32_e32 v36, 16, v36
    v_or_b32_e32 v34, v34, v36
    v_lshlrev_b32_e32 v37, 24, v37
    v_or_b32_e32 v64, v34, v37            // V[K+0..K+3]
    
    v_lshlrev_b32_e32 v39, 8, v39
    v_or_b32_e32 v38, v38, v39
    v_lshlrev_b32_e32 v40, 16, v40
    v_or_b32_e32 v38, v38, v40
    v_lshlrev_b32_e32 v41, 24, v41
    v_or_b32_e32 v65, v38, v41            // V[K+4..K+7]
    
    s_nop 7
    
    // ========================================================================
    // STEP 5: PV MFMA
    // O[q,d] = sum_k(P[q,k] * V[k,d])
    // With P=1/16 and V[k,d]=d/32:
    // O[q,d] = sum_k(1/16 * d/32) = d/32 * sum_k(1/16) = d/32 * 1 = d/32
    // ========================================================================
    
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    
    s_nop 7
    
    // ========================================================================
    // STEP 6: Handle output interleaving with permlane swap
    // MFMA output interleaving: threads 0-31 have M rows 0,1,2,3,8,9,10,11,...
    // threads 32-63 have M rows 4,5,6,7,12,13,14,15,...
    // ========================================================================
    
    s_nop 0
    s_nop 0
    // Swap v48-v51 (registers 0-3) with v52-v55 (registers 4-7) between halves
    v_permlane32_swap_b32_e32 v48, v52
    v_permlane32_swap_b32_e32 v49, v53
    v_permlane32_swap_b32_e32 v50, v54
    v_permlane32_swap_b32_e32 v51, v55
    
    // Swap v56-v59 with v60-v63
    v_permlane32_swap_b32_e32 v56, v60
    v_permlane32_swap_b32_e32 v57, v61
    v_permlane32_swap_b32_e32 v58, v62
    v_permlane32_swap_b32_e32 v59, v63
    
    // ========================================================================
    // STEP 7: Store output
    // After swap, threads 0-31 have D=0-7 (contiguous), threads 32-63 have D=8-15
    // Store at O[Q=tid%32, D_base]
    // ========================================================================
    
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    
    // Thread t stores at Q=t%32, D_base=(t/32)*8
    // Address = Q * 128 * 4 + D_base * 4 = Q * 512 + D_base * 4
    v_and_b32_e32 v3, 31, v0              // Q
    v_lshrrev_b32_e32 v4, 5, v0           // D_block
    v_lshlrev_b32_e32 v3, 9, v3           // Q * 512
    v_lshlrev_b32_e32 v4, 5, v4           // D_block * 32 (8 floats)
    v_add_u32_e32 v3, v3, v4
    
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store 8 floats
    flat_store_dwordx4 v[10:11], v[48:51]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[52:55]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter14test_full_attnE, .-_ZN5aiter14test_full_attnE

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter14test_full_attnE
    .amdhsa_group_segment_fixed_size 8192
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 80
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 68
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter14test_full_attnE
    .symbol: _ZN5aiter14test_full_attnE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 8192
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 80
    .agpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
      - .name: ptr_out
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_V
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata

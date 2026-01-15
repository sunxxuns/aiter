// FP8 Full Attention (Step 3): O = softmax(Q @ K^T / sqrt(d)) @ V
// 256 threads (4 waves), Q/K/V are 32×128 tiles

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set SCALE, 0x3e028f5c              // log2(e) / sqrt(128) = 0.12754

.text
.globl _ZN5aiter15fwd_fp8_swizzleE
.p2align 8
.type _ZN5aiter15fwd_fp8_swizzleE,@function

_ZN5aiter15fwd_fp8_swizzleE:
    s_mov_b64 exec, -1
    
    // Args: O_ptr, Q_ptr, K_ptr, V_ptr
    s_load_dwordx4 s[4:7], s[0:1], 0       // O_ptr, Q_ptr
    s_load_dwordx4 s[12:15], s[0:1], 16    // K_ptr, V_ptr
    s_waitcnt lgkmcnt(0)
    
    v_and_b32_e32 v60, 63, v0              // lane_id
    v_lshrrev_b32_e32 v61, 6, v0           // wave_id
    
    s_mov_b32 s2, SCALE
    
    // ========================================================================
    // LOAD Q[32×128] TO LDS (offset 0)
    // ========================================================================
    v_lshlrev_b32_e32 v1, 4, v60           // lane * 16
    v_lshlrev_b32_e32 v2, 10, v61          // wave * 1024
    v_add_u32_e32 v4, v1, v2
    
    v_mov_b32_e32 v10, s6                  // Q_ptr
    v_mov_b32_e32 v11, s7
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v4, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // LOAD K[32×128] TO LDS (offset 4096)
    // ========================================================================
    v_mov_b32_e32 v10, s12                 // K_ptr
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    v_add_u32_e32 v5, 4096, v4
    
    flat_load_dwordx4 v[24:27], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v5, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // COMPUTE QK MFMA
    // ========================================================================
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v63, v3, v4              // mfma_row
    
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v64, 0, 8, vcc       // k_half
    
    v_lshlrev_b32_e32 v10, 7, v63
    v_add_u32_e32 v10, v10, v64
    
    v_mov_b32_e32 v70, v10                 // Q addr
    v_add_u32_e32 v71, 4096, v10           // K addr
    
    // Clear accumulators
    .irp i, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        v_mov_b32_e32 v\i, 0
    .endr
    
    // QK MFMA (8 iterations for HD=128)
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
    // SOFTMAX
    // ========================================================================
    // Row max
    v_max_f32_e32 v20, v0, v1
    v_max_f32_e32 v20, v20, v2
    v_max_f32_e32 v20, v20, v3
    v_max_f32_e32 v20, v20, v4
    v_max_f32_e32 v20, v20, v5
    v_max_f32_e32 v20, v20, v6
    v_max_f32_e32 v20, v20, v7
    v_max_f32_e32 v20, v20, v8
    v_max_f32_e32 v20, v20, v9
    v_max_f32_e32 v20, v20, v10
    v_max_f32_e32 v20, v20, v11
    v_max_f32_e32 v20, v20, v12
    v_max_f32_e32 v20, v20, v13
    v_max_f32_e32 v20, v20, v14
    v_max_f32_e32 v20, v20, v15
    
    v_mov_b32_e32 v21, v20
    s_nop 1
    v_permlane32_swap_b32_e32 v21, v20
    v_max_f32_e32 v20, v20, v21
    
    // exp((S - max) * scale)
    v_mul_f32_e32 v21, s2, v20
    
    v_fma_f32 v0, v0, s2, -v21
    v_fma_f32 v1, v1, s2, -v21
    v_fma_f32 v2, v2, s2, -v21
    v_fma_f32 v3, v3, s2, -v21
    v_fma_f32 v4, v4, s2, -v21
    v_fma_f32 v5, v5, s2, -v21
    v_fma_f32 v6, v6, s2, -v21
    v_fma_f32 v7, v7, s2, -v21
    v_fma_f32 v8, v8, s2, -v21
    v_fma_f32 v9, v9, s2, -v21
    v_fma_f32 v10, v10, s2, -v21
    v_fma_f32 v11, v11, s2, -v21
    v_fma_f32 v12, v12, s2, -v21
    v_fma_f32 v13, v13, s2, -v21
    v_fma_f32 v14, v14, s2, -v21
    v_fma_f32 v15, v15, s2, -v21
    
    v_exp_f32_e32 v0, v0
    v_exp_f32_e32 v1, v1
    v_exp_f32_e32 v2, v2
    v_exp_f32_e32 v3, v3
    v_exp_f32_e32 v4, v4
    v_exp_f32_e32 v5, v5
    v_exp_f32_e32 v6, v6
    v_exp_f32_e32 v7, v7
    v_exp_f32_e32 v8, v8
    v_exp_f32_e32 v9, v9
    v_exp_f32_e32 v10, v10
    v_exp_f32_e32 v11, v11
    v_exp_f32_e32 v12, v12
    v_exp_f32_e32 v13, v13
    v_exp_f32_e32 v14, v14
    v_exp_f32_e32 v15, v15
    s_nop 7
    s_nop 7
    s_nop 7
    
    // Row sum
    v_add_f32_e32 v22, v0, v1
    v_add_f32_e32 v22, v22, v2
    v_add_f32_e32 v22, v22, v3
    v_add_f32_e32 v22, v22, v4
    v_add_f32_e32 v22, v22, v5
    v_add_f32_e32 v22, v22, v6
    v_add_f32_e32 v22, v22, v7
    v_add_f32_e32 v22, v22, v8
    v_add_f32_e32 v22, v22, v9
    v_add_f32_e32 v22, v22, v10
    v_add_f32_e32 v22, v22, v11
    v_add_f32_e32 v22, v22, v12
    v_add_f32_e32 v22, v22, v13
    v_add_f32_e32 v22, v22, v14
    v_add_f32_e32 v22, v22, v15
    
    v_mov_b32_e32 v23, v22
    s_nop 1
    v_permlane32_swap_b32_e32 v23, v22
    v_add_f32_e32 v22, v22, v23
    
    // Normalize
    v_rcp_f32_e32 v22, v22
    s_nop 3
    
    v_mul_f32_e32 v0, v0, v22
    v_mul_f32_e32 v1, v1, v22
    v_mul_f32_e32 v2, v2, v22
    v_mul_f32_e32 v3, v3, v22
    v_mul_f32_e32 v4, v4, v22
    v_mul_f32_e32 v5, v5, v22
    v_mul_f32_e32 v6, v6, v22
    v_mul_f32_e32 v7, v7, v22
    v_mul_f32_e32 v8, v8, v22
    v_mul_f32_e32 v9, v9, v22
    v_mul_f32_e32 v10, v10, v22
    v_mul_f32_e32 v11, v11, v22
    v_mul_f32_e32 v12, v12, v22
    v_mul_f32_e32 v13, v13, v22
    v_mul_f32_e32 v14, v14, v22
    v_mul_f32_e32 v15, v15, v22
    
    // P is now in v[0:15], normalized softmax
    
    // ========================================================================
    // STORE P TO LDS (offset 8192) - SAME PATTERN AS WORKING KERNEL
    // ========================================================================
    s_barrier
    
    // P layout: P[col, k] stored as P[col * 32 + k]
    // col = lane & 31, k_base = (lane >> 5) * 4
    // v0-v15 contain values for k positions based on MFMA output
    v_and_b32_e32 v24, 31, v60            // col = lane & 31
    v_lshrrev_b32_e32 v25, 5, v60         // half = lane >> 5
    v_lshlrev_b32_e32 v25, 2, v25         // half * 4
    
    // Store P using working kernel's STORE_P pattern
    // addr = 8192 + (col * 32 + k_8_group * 8 + k_mod4 + half*4) * 4
    .macro STORE_P_256T vreg, k_mod4, k_8_group
        v_mov_b32_e32 v26, \k_mod4
        v_add_u32_e32 v26, v26, v25       // + half * 4
        v_add_u32_e32 v26, \k_8_group * 8, v26
        v_lshlrev_b32_e32 v27, 5, v24     // col * 32
        v_add_u32_e32 v27, v27, v26
        v_lshlrev_b32_e32 v27, 2, v27     // * 4 (bytes)
        v_add_u32_e32 v27, 8192, v27
        ds_write_b32 v27, \vreg
    .endm
    
    STORE_P_256T v0, 0, 0
    STORE_P_256T v1, 1, 0
    STORE_P_256T v2, 2, 0
    STORE_P_256T v3, 3, 0
    STORE_P_256T v4, 0, 1
    STORE_P_256T v5, 1, 1
    STORE_P_256T v6, 2, 1
    STORE_P_256T v7, 3, 1
    STORE_P_256T v8, 0, 2
    STORE_P_256T v9, 1, 2
    STORE_P_256T v10, 2, 2
    STORE_P_256T v11, 3, 2
    STORE_P_256T v12, 0, 3
    STORE_P_256T v13, 1, 3
    STORE_P_256T v14, 2, 3
    STORE_P_256T v15, 3, 3
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // Load V[32×128] to LDS (offset 12288, after P)
    // Reuse Q loading pattern
    v_lshlrev_b32_e32 v1, 4, v60
    v_lshlrev_b32_e32 v2, 10, v61
    v_add_u32_e32 v4, v1, v2
    
    v_mov_b32_e32 v10, s14                // V_ptr
    v_mov_b32_e32 v11, s15
    v_add_co_u32_e32 v10, vcc, v4, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    v_add_u32_e32 v5, 12288, v4           // V at LDS offset 12288
    
    flat_load_dwordx4 v[24:27], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v5, v[24:27]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // PV MFMA: O = P @ V
    // P at offset 8192, V at offset 12288
    // O output: v[80:95] for HD tile 0 (cols 0-31)
    // ========================================================================
    
    // Initialize O accumulators
    .irp i, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95
        v_mov_b32_e32 v\i, 0
    .endr
    
    // Read P and V, compute MFMA
    // For simplicity, just do one HD tile (cols 0-31 of V)
    
    // Read P row - P[col * 32 + k] where col = mfma_row, k = 0..15 (first pass)
    // addr = 8192 + (col * 32 + k) * 4
    v_and_b32_e32 v24, 31, v60
    v_lshrrev_b32_e32 v25, 5, v60
    v_lshlrev_b32_e32 v26, 7, v63         // mfma_row * 128 (col * 32 * 4)
    v_lshlrev_b32_e32 v27, 2, v64         // k_half * 4 (bytes per F32)
    v_add_u32_e32 v26, v26, v27
    v_add_u32_e32 v26, 8192, v26          // P addr base
    
    // Read 8 F32 values from P (k=0..7 for first MFMA in this pass)
    ds_read_b64 v[28:29], v26
    v_add_u32_e32 v27, 8, v26
    ds_read_b64 v[30:31], v27
    v_add_u32_e32 v27, 16, v26
    ds_read_b64 v[32:33], v27
    v_add_u32_e32 v27, 24, v26
    ds_read_b64 v[34:35], v27
    s_waitcnt lgkmcnt(0)
    
    // Convert P to FP8 (8 F32 → 8 FP8 in 2 dwords)
    v_cvt_pk_fp8_f32 v36, v28, v29
    v_and_b32_e32 v36, 0xFFFF, v36
    v_cvt_pk_fp8_f32 v37, v30, v31
    v_lshlrev_b32_e32 v37, 16, v37
    v_and_b32_e32 v37, 0xFFFF0000, v37
    v_or_b32_e32 v36, v36, v37            // v36 = first 4 FP8
    
    v_cvt_pk_fp8_f32 v37, v32, v33
    v_and_b32_e32 v37, 0xFFFF, v37
    v_cvt_pk_fp8_f32 v38, v34, v35
    v_lshlrev_b32_e32 v38, 16, v38
    v_and_b32_e32 v38, 0xFFFF0000, v38
    v_or_b32_e32 v37, v37, v38            // v37 = next 4 FP8
    
    // Read V column (8 bytes from 8 rows, stride 128)
    // V is at LDS offset 12288, col 0
    v_lshlrev_b32_e32 v38, 10, v25        // half * 1024
    v_add_u32_e32 v38, v38, v24           // + lane & 31
    v_add_u32_e32 v38, 12288, v38         // V base
    
    // Read 8 FP8 bytes from V column
    ds_read_u8 v40, v38
    v_add_u32_e32 v39, 128, v38
    ds_read_u8 v41, v39
    v_add_u32_e32 v39, 256, v38
    ds_read_u8 v42, v39
    v_add_u32_e32 v39, 384, v38
    ds_read_u8 v43, v39
    v_add_u32_e32 v39, 512, v38
    ds_read_u8 v44, v39
    v_add_u32_e32 v39, 640, v38
    ds_read_u8 v45, v39
    v_add_u32_e32 v39, 768, v38
    ds_read_u8 v46, v39
    v_add_u32_e32 v39, 896, v38
    ds_read_u8 v47, v39
    s_waitcnt lgkmcnt(0)
    
    // Pack V bytes into 2 dwords
    v_lshlrev_b32_e32 v41, 8, v41
    v_or_b32_e32 v40, v40, v41
    v_lshlrev_b32_e32 v42, 16, v42
    v_or_b32_e32 v40, v40, v42
    v_lshlrev_b32_e32 v43, 24, v43
    v_or_b32_e32 v48, v40, v43            // v48 = first 4 FP8
    
    v_lshlrev_b32_e32 v45, 8, v45
    v_or_b32_e32 v44, v44, v45
    v_lshlrev_b32_e32 v46, 16, v46
    v_or_b32_e32 v44, v44, v46
    v_lshlrev_b32_e32 v47, 24, v47
    v_or_b32_e32 v49, v44, v47            // v49 = next 4 FP8
    
    // PV MFMA (one pass for k=0..15)
    v_accvgpr_write_b32 a0, v36
    v_accvgpr_write_b32 a1, v37
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[48:49], v[80:95]
    s_nop 15
    
    // Second PV pass for k=16..31
    v_add_u32_e32 v26, 32, v26            // Advance P addr
    ds_read_b64 v[28:29], v26
    v_add_u32_e32 v27, 8, v26
    ds_read_b64 v[30:31], v27
    v_add_u32_e32 v27, 16, v26
    ds_read_b64 v[32:33], v27
    v_add_u32_e32 v27, 24, v26
    ds_read_b64 v[34:35], v27
    s_waitcnt lgkmcnt(0)
    
    v_cvt_pk_fp8_f32 v36, v28, v29
    v_and_b32_e32 v36, 0xFFFF, v36
    v_cvt_pk_fp8_f32 v37, v30, v31
    v_lshlrev_b32_e32 v37, 16, v37
    v_and_b32_e32 v37, 0xFFFF0000, v37
    v_or_b32_e32 v36, v36, v37
    
    v_cvt_pk_fp8_f32 v37, v32, v33
    v_and_b32_e32 v37, 0xFFFF, v37
    v_cvt_pk_fp8_f32 v38, v34, v35
    v_lshlrev_b32_e32 v38, 16, v38
    v_and_b32_e32 v38, 0xFFFF0000, v38
    v_or_b32_e32 v37, v37, v38
    
    // V for k=16..31 (offset +2048 = 16 rows * 128)
    v_add_u32_e32 v38, 2048, v38
    ds_read_u8 v40, v38
    v_add_u32_e32 v39, 128, v38
    ds_read_u8 v41, v39
    v_add_u32_e32 v39, 256, v38
    ds_read_u8 v42, v39
    v_add_u32_e32 v39, 384, v38
    ds_read_u8 v43, v39
    v_add_u32_e32 v39, 512, v38
    ds_read_u8 v44, v39
    v_add_u32_e32 v39, 640, v38
    ds_read_u8 v45, v39
    v_add_u32_e32 v39, 768, v38
    ds_read_u8 v46, v39
    v_add_u32_e32 v39, 896, v38
    ds_read_u8 v47, v39
    s_waitcnt lgkmcnt(0)
    
    v_lshlrev_b32_e32 v41, 8, v41
    v_or_b32_e32 v40, v40, v41
    v_lshlrev_b32_e32 v42, 16, v42
    v_or_b32_e32 v40, v40, v42
    v_lshlrev_b32_e32 v43, 24, v43
    v_or_b32_e32 v48, v40, v43
    
    v_lshlrev_b32_e32 v45, 8, v45
    v_or_b32_e32 v44, v44, v45
    v_lshlrev_b32_e32 v46, 16, v46
    v_or_b32_e32 v44, v44, v46
    v_lshlrev_b32_e32 v47, 24, v47
    v_or_b32_e32 v49, v44, v47
    
    v_accvgpr_write_b32 a0, v36
    v_accvgpr_write_b32 a1, v37
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], a[0:1], v[48:49], v[80:95]
    s_nop 15
    
    // O is now in v[80:95] (16 F32 values per lane for HD tile 0)
    
    // ========================================================================
    // STORE O OUTPUT
    // ========================================================================
    v_mov_b32_e32 v40, s4                 // O_ptr
    v_mov_b32_e32 v41, s5
    v_lshlrev_b32_e32 v42, 6, v60         // lane * 64
    v_add_co_u32_e32 v40, vcc, v42, v40
    v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
    
    flat_store_dwordx4 v[40:41], v[80:83]
    v_add_co_u32_e32 v42, vcc, 16, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[84:87]
    v_add_co_u32_e32 v42, vcc, 32, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[88:91]
    v_add_co_u32_e32 v42, vcc, 48, v40
    v_addc_co_u32_e32 v43, vcc, 0, v41, vcc
    flat_store_dwordx4 v[42:43], v[92:95]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter15fwd_fp8_swizzleE
    .amdhsa_group_segment_fixed_size 20480
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 32
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 144
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 144
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter15fwd_fp8_swizzleE
    .symbol: _ZN5aiter15fwd_fp8_swizzleE.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 20480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 144
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V_ptr, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

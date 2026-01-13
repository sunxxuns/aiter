// FP8 Flash Attention - 128 Q rows, 256 threads (4 waves)
// Direct port of BF16 structure:
// - 4 waves, each processing 32 Q rows
// - Output: O[128×128]
// - 4 HD tiles per wave for full HD=128 coverage
//
// Wave mapping:
//   wave 0: Q rows 0-31   → O rows 0-31
//   wave 1: Q rows 32-63  → O rows 32-63
//   wave 2: Q rows 64-95  → O rows 64-95
//   wave 3: Q rows 96-127 → O rows 96-127

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13fwd_fp8_128qE
.p2align 8
.type _ZN5aiter13fwd_fp8_128qE,@function

_ZN5aiter13fwd_fp8_128qE:
    s_mov_b64 exec, -1
    
    // ========================================================================
    // LOAD KERNEL ARGS
    // ========================================================================
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O output [128×128] F32
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q [128×128] FP8
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K [seq×128] FP8
    s_load_dwordx2 s[16:17], s[0:1], 0x18  // V [seq×128] FP8
    s_load_dword s24, s[0:1], 0x20         // seq_len
    
    // ========================================================================
    // THREAD SETUP (BF16 pattern lines 44-54)
    // ========================================================================
    // tid = v0 (0-255)
    // wave_id = tid >> 6 (0-3)
    // lane_id = tid & 63 (0-63)
    v_lshrrev_b32_e32 v61, 6, v0          // wave_id (0-3)
    v_and_b32_e32 v60, 63, v0             // lane_id (0-63)
    v_readfirstlane_b32 s28, v61          // s28 = wave_id (SGPR) - don't clobber s5!
    
    // Constants
    s_mov_b32 s2, 0x3e028f5c              // log2(e) / sqrt(128)
    s_mov_b32 s3, 0xff800000              // -infinity
    
    s_waitcnt lgkmcnt(0)
    
    // Number of K-tiles: (seq_len + 31) / 32
    s_add_i32 s25, s24, 31
    s_lshr_b32 s25, s25, 5
    
    // ========================================================================
    // BUFFER DESCRIPTORS
    // ========================================================================
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s18, -1
    s_mov_b32 s19, 0x20000
    
    // ========================================================================
    // MFMA ROW MAPPING (same for all waves, using lane_id)
    // ========================================================================
    v_and_b32_e32 v1, 3, v60
    v_lshrrev_b32_e32 v2, 3, v60
    v_and_b32_e32 v2, 3, v2
    v_lshlrev_b32_e32 v2, 2, v2
    v_add_u32_e32 v3, v1, v2
    v_lshrrev_b32_e32 v4, 2, v60
    v_and_b32_e32 v4, 1, v4
    v_lshlrev_b32_e32 v4, 4, v4
    v_add_u32_e32 v62, v3, v4             // mfma_row (0-31 within wave's 32 rows)
    
    // k_base: 0 for lanes 0-31, 8 for lanes 32-63
    v_mov_b32_e32 v63, 0
    v_mov_b32_e32 v4, 8
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e32 v63, v63, v4, vcc
    
    // ========================================================================
    // LOAD Q TO LDS (all 128 rows, cooperative)
    // Each thread loads 16 bytes: 256 threads × 16 = 4KB per iteration
    // Q[128×128] = 16KB total, need 4 iterations
    // ========================================================================
    
    // Q region starts at LDS offset 0
    // Q layout: row-major, Q[row, col] at LDS[row * 128 + col]
    
    // Iteration 0: Q rows 0-31 (threads load different 16B chunks)
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    // Iteration 1: Q rows 32-63
    v_add_u32_e32 v1, 4096, v1            // +4KB
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    // Iteration 2: Q rows 64-95
    v_add_u32_e32 v1, 4096, v1
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    // Iteration 3: Q rows 96-127
    v_add_u32_e32 v1, 4096, v1
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ========================================================================
    // Q LDS ADDRESS FOR THIS WAVE
    // wave_id determines which 32 Q rows this wave processes
    // Q_base = wave_id * 32 * 128 = wave_id * 4096
    // ========================================================================
    s_mul_i32 s40, s28, 4096              // wave_id * 4096
    v_lshlrev_b32_e32 v70, 7, v62         // mfma_row * 128
    v_add_u32_e32 v70, v70, v63           // + k_base
    v_add_u32_e32 v70, s40, v70           // + wave's Q base
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE
    // ========================================================================
    v_mov_b32_e32 v64, s3                 // running_max = -inf
    v_mov_b32_e32 v65, 0                  // running_sum = 0
    
    // Initialize O accumulators (4 HD tiles × 16 values = 64 VGPRs)
    // HD tile 0: v[80:95], HD tile 1: v[96:111], HD tile 2: v[112:127], HD tile 3: v[128:143]
    .irp i, 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
        v_mov_b32_e32 v\i, 0
    .endr
    .irp i, 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143
        v_mov_b32_e32 v\i, 0
    .endr
    
    // K-tile loop counter
    s_mov_b32 s26, 0                       // tile_idx = 0
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
K_TILE_LOOP:
    
    // --------------------------------------------------------------------
    // COOPERATIVE K LOAD (256 threads load 32×128 = 4KB)
    // K region at LDS offset 16384 (after Q)
    // --------------------------------------------------------------------
    s_lshl_b32 s27, s26, 12               // tile_offset = tile_idx * 4096
    
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16
    v_mov_b32_e32 v2, s27
    v_add_u32_e32 v1, v1, v2
    
    v_mov_b32_e32 v10, s12
    v_mov_b32_e32 v11, s13
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Restore tid*16 for LDS address
    v_lshlrev_b32_e32 v1, 4, v0
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v1, 16384, v1           // K region starts at 16384
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // --------------------------------------------------------------------
    // QK MFMA: S = Q @ K^T (each wave computes its own 32×32 S)
    // 8 MFMAs for HD=128
    // --------------------------------------------------------------------
    
    // K LDS address: 16384 + mfma_row * 128 + k_base
    v_lshlrev_b32_e32 v71, 7, v62
    v_add_u32_e32 v71, v71, v63
    v_add_u32_e32 v71, 16384, v71
    
    // Clear S accumulators
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
        v_add_u32_e32 v72, \k_off, v71
        ds_read_b64 v[30:31], v72
        v_add_u32_e32 v73, \k_off, v70
        ds_read_b64 v[32:33], v73
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
        s_nop 7
    .endr
    
    s_nop 15
    
    // S is now in v[0:15]
    
    // --------------------------------------------------------------------
    // SOFTMAX (online algorithm)
    // Scale, find max, update running state, compute P = exp(S - max)
    // --------------------------------------------------------------------
    
    // Scale by log2(e)/sqrt(d)
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        v_mul_f32_e32 v\i, s2, v\i
    .endr
    
    // Find tile_max
    v_max_f32_e32 v16, v0, v1
    v_max_f32_e32 v16, v16, v2
    v_max_f32_e32 v16, v16, v3
    v_max_f32_e32 v16, v16, v4
    v_max_f32_e32 v16, v16, v5
    v_max_f32_e32 v16, v16, v6
    v_max_f32_e32 v16, v16, v7
    v_max_f32_e32 v16, v16, v8
    v_max_f32_e32 v16, v16, v9
    v_max_f32_e32 v16, v16, v10
    v_max_f32_e32 v16, v16, v11
    v_max_f32_e32 v16, v16, v12
    v_max_f32_e32 v16, v16, v13
    v_max_f32_e32 v16, v16, v14
    v_max_f32_e32 v16, v16, v15
    
    // new_max = max(running_max, tile_max)
    v_max_f32_e32 v17, v64, v16
    
    // correction = exp2(running_max - new_max)
    v_sub_f32_e32 v18, v64, v17
    v_exp_f32_e32 v18, v18
    
    // Update running_sum *= correction
    v_mul_f32_e32 v65, v65, v18
    
    // Scale O accumulators by correction
    .irp i, 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
        v_mul_f32_e32 v\i, v\i, v18
    .endr
    .irp i, 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111
        v_mul_f32_e32 v\i, v\i, v18
    .endr
    .irp i, 112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
        v_mul_f32_e32 v\i, v\i, v18
    .endr
    .irp i, 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143
        v_mul_f32_e32 v\i, v\i, v18
    .endr
    
    // P = exp2(S - new_max)
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        v_sub_f32_e32 v\i, v\i, v17
        v_exp_f32_e32 v\i, v\i
    .endr
    
    // tile_sum = sum(P)
    v_add_f32_e32 v19, v0, v1
    v_add_f32_e32 v19, v19, v2
    v_add_f32_e32 v19, v19, v3
    v_add_f32_e32 v19, v19, v4
    v_add_f32_e32 v19, v19, v5
    v_add_f32_e32 v19, v19, v6
    v_add_f32_e32 v19, v19, v7
    v_add_f32_e32 v19, v19, v8
    v_add_f32_e32 v19, v19, v9
    v_add_f32_e32 v19, v19, v10
    v_add_f32_e32 v19, v19, v11
    v_add_f32_e32 v19, v19, v12
    v_add_f32_e32 v19, v19, v13
    v_add_f32_e32 v19, v19, v14
    v_add_f32_e32 v19, v19, v15
    
    // running_sum += tile_sum
    v_add_f32_e32 v65, v65, v19
    
    // running_max = new_max
    v_mov_b32_e32 v64, v17
    
    // Convert P to FP8 for PV MFMA
    v_cvt_pk_fp8_f32 v40, v0, v1
    v_cvt_pk_fp8_f32 v41, v2, v3
    v_cvt_pk_fp8_f32 v42, v4, v5
    v_cvt_pk_fp8_f32 v43, v6, v7
    v_cvt_pk_fp8_f32 v44, v8, v9
    v_cvt_pk_fp8_f32 v45, v10, v11
    v_cvt_pk_fp8_f32 v46, v12, v13
    v_cvt_pk_fp8_f32 v47, v14, v15
    
    // --------------------------------------------------------------------
    // COOPERATIVE V LOAD (256 threads load 32×128 = 4KB)
    // V region at LDS offset 20480 (after Q and K)
    // --------------------------------------------------------------------
    
    v_lshlrev_b32_e32 v1, 4, v0           // tid * 16 (restore tid from v60 calc)
    // Actually v0 was clobbered, need to recalculate
    v_lshlrev_b32_e32 v1, 6, v61          // wave_id * 64
    v_add_u32_e32 v1, v60, v1             // + lane_id = tid
    v_lshlrev_b32_e32 v1, 4, v1           // tid * 16
    
    v_mov_b32_e32 v2, s27
    v_add_u32_e32 v2, v1, v2              // + tile_offset
    
    v_mov_b32_e32 v10, s16
    v_mov_b32_e32 v11, s17
    v_add_co_u32_e32 v10, vcc, v2, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[20:23], v[10:11]
    s_waitcnt vmcnt(0)
    
    v_add_u32_e32 v1, 20480, v1
    ds_write_b128 v1, v[20:23]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // --------------------------------------------------------------------
    // PV MFMA: O += P @ V (4 HD tiles)
    // All waves use same V data, but different P (each wave's S was different)
    // --------------------------------------------------------------------
    
    // V LDS address: 20480 + mfma_row * 128 + k_base
    v_lshlrev_b32_e32 v74, 7, v62
    v_add_u32_e32 v74, v74, v63
    v_add_u32_e32 v74, 20480, v74
    
    // HD tile 0 (cols 0-31): v[80:95]
    ds_read_b64 v[30:31], v74
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[80:95], v[40:41], v[30:31], v[80:95]
    s_nop 7
    
    // HD tile 1 (cols 32-63): v[96:111]
    v_add_u32_e32 v75, 32, v74
    ds_read_b64 v[30:31], v75
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[96:111], v[42:43], v[30:31], v[96:111]
    s_nop 7
    
    // HD tile 2 (cols 64-95): v[112:127]
    v_add_u32_e32 v75, 64, v74
    ds_read_b64 v[30:31], v75
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[112:127], v[44:45], v[30:31], v[112:127]
    s_nop 7
    
    // HD tile 3 (cols 96-127): v[128:143]
    v_add_u32_e32 v75, 96, v74
    ds_read_b64 v[30:31], v75
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[128:143], v[46:47], v[30:31], v[128:143]
    s_nop 7
    
    // --------------------------------------------------------------------
    // LOOP INCREMENT
    // --------------------------------------------------------------------
    s_add_i32 s26, s26, 1
    s_cmp_lt_i32 s26, s25
    s_cbranch_scc1 K_TILE_LOOP
    
    // ========================================================================
    // FINALIZE: O = O / running_sum
    // ========================================================================
    v_rcp_f32_e32 v65, v65                // 1/running_sum
    
    .irp i, 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
        v_mul_f32_e32 v\i, v\i, v65
    .endr
    .irp i, 96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111
        v_mul_f32_e32 v\i, v\i, v65
    .endr
    .irp i, 112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127
        v_mul_f32_e32 v\i, v\i, v65
    .endr
    .irp i, 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143
        v_mul_f32_e32 v\i, v\i, v65
    .endr
    
    // ========================================================================
    // OUTPUT: Each wave writes its 32 rows
    // O offset: wave_id * 32 * 128 * 4 = wave_id * 16384 bytes
    // ========================================================================
    
    s_mul_i32 s40, s28, 16384             // wave_id * 16384
    
    // Per-lane offset within wave's output region
    // mfma_row * 512 (32 cols * 4 tiles * 4 bytes)
    v_lshlrev_b32_e32 v1, 9, v62
    v_add_u32_e32 v1, s40, v1
    
    v_mov_b32_e32 v10, s4
    v_mov_b32_e32 v11, s5
    v_add_co_u32_e32 v10, vcc, v1, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store HD tile 0 (v[80:95])
    flat_store_dwordx4 v[10:11], v[80:83]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[84:87]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[88:91]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[92:95]
    
    // Store HD tile 1 (v[96:111]) at +128 bytes
    v_add_co_u32_e32 v10, vcc, 128, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[96:99]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[100:103]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[104:107]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[108:111]
    
    // Store HD tile 2 (v[112:127]) at +256 bytes
    v_add_co_u32_e32 v10, vcc, 128, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[112:115]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[116:119]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[120:123]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[124:127]
    
    // Store HD tile 3 (v[128:143]) at +384 bytes
    v_add_co_u32_e32 v10, vcc, 128, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[128:131]
    v_add_co_u32_e32 v12, vcc, 16, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[132:135]
    v_add_co_u32_e32 v12, vcc, 32, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[136:139]
    v_add_co_u32_e32 v12, vcc, 48, v10
    v_addc_co_u32_e32 v13, vcc, 0, v11, vcc
    flat_store_dwordx4 v[12:13], v[140:143]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13fwd_fp8_128qE
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 40
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 160
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 160
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter13fwd_fp8_128qE
    .symbol: _ZN5aiter13fwd_fp8_128qE.kd
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 160
    .max_flat_workgroup_size: 256
    .args:
      - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
      - {.name: V, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global}
      - {.name: seq_len, .size: 4, .offset: 32, .value_kind: by_value}
...
.end_amdgpu_metadata

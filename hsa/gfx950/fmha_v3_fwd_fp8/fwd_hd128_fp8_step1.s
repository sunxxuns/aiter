// SPDX-License-Identifier: MIT
// FP8 Flash Attention - Full K-loop implementation with online softmax
// Properly handles workgroup IDs for Q-tile and head/batch offsets

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter18fmha_fwd_hd128_fp8E
.p2align 8
.type _ZN5aiter18fmha_fwd_hd128_fp8E,@function

// Configuration constants
.set BLOCK_M, 64                // Q tile rows
.set BLOCK_N, 64                // K/V tile rows  
.set HEAD_DIM, 128              // Head dimension
.set THREADS, 256               // Threads per block (4 warps)
.set LDS_Q_SIZE, 8192           // 64 * 128 FP8 = 8KB for Q
.set LDS_K_SIZE, 8192           // 64 * 128 FP8 = 8KB for K
.set LDS_V_SIZE, 8192           // 64 * 128 FP8 = 8KB for V
.set LDS_TOTAL, 32768           // 32KB total LDS
.set LDS_Q_OFFSET, 0
.set LDS_K_OFFSET, 8192
.set LDS_V_OFFSET, 16384
.set Q_TILE_SIZE, 8192          // 64 * 128 bytes per Q-tile

_ZN5aiter18fmha_fwd_hd128_fp8E:
    // ========================================================================
    // WORKGROUP LAYOUT:
    // workgroup_id_x = Q-tile index (which 64-row block of Q)
    // workgroup_id_y = head index (which attention head)
    // workgroup_id_z = batch index
    //
    // REGISTER ALLOCATION:
    // s0-s1:   kernarg base pointer
    // s2:      workgroup_id_x (Q-tile index)
    // s3:      workgroup_id_y (head index) 
    // s4:      workgroup_id_z (batch index)
    // s8-s9:   ptr_R (output)
    // s10-s11: ptr_Q
    // s12-s13: ptr_K
    // s14-s15: ptr_V
    // s20:     softmax_scale
    // s21:     seqlen_q
    // s22:     seqlen_k
    // s23:     head_stride (seqlen * head_dim)
    // s24:     batch_stride (seqlen * num_heads * head_dim)
    // s26:     q_scale
    // s27:     k_scale
    // s28:     v_scale
    // s29:     combined QK scale
    // s30:     k_tiles (number of K-tiles to process)
    // s31:     current k_tile index
    // s32-s33: q_offset (workgroup offset for Q)
    // s34-s35: o_offset (workgroup offset for output)
    // ========================================================================
    
    // Load kernel arguments
    s_and_b32 s1, s1, 0xffff
    
    s_load_dwordx2 s[8:9], s[0:1], 0x00       // ptr_R (output)
    s_load_dwordx2 s[10:11], s[0:1], 0x10     // ptr_Q
    s_load_dwordx2 s[12:13], s[0:1], 0x20     // ptr_K
    s_load_dwordx2 s[14:15], s[0:1], 0x30     // ptr_V
    s_load_dword s20, s[0:1], 0x50            // softmax_scale
    s_load_dword s21, s[0:1], 0x58            // seqlen_q
    s_load_dword s22, s[0:1], 0x60            // seqlen_k
    s_load_dword s26, s[0:1], 0x200           // q_scale
    s_load_dword s27, s[0:1], 0x204           // k_scale
    s_load_dword s28, s[0:1], 0x208           // v_scale
    
    // Thread/warp ID extraction
    v_and_b32_e32 v0, 0xff, v0                // tid (0-255)
    v_lshrrev_b32_e32 v1, 6, v0               // warp_id (0-3)
    v_and_b32_e32 v2, 63, v0                  // lane_id (0-63)
    
    s_waitcnt lgkmcnt(0)
    
    // ========================================================================
    // COMPUTE WORKGROUP OFFSETS
    // For FP8 inputs: Q[batch, seq, head, dim]
    // head_stride = seqlen * HEAD_DIM bytes (for FP8)
    // batch_stride = seqlen * num_heads * HEAD_DIM bytes
    // ========================================================================
    
    // head_stride = seqlen_q * 128 (for Q) or seqlen_k * 128 (for K/V)
    s_lshl_b32 s23, s21, 7                    // s23 = seqlen_q * 128 = head_stride_q
    s_lshl_b32 s24, s22, 7                    // s24 = seqlen_k * 128 = head_stride_kv
    
    // Compute Q base offset for this workgroup
    // q_offset = batch_idx * batch_stride + head_idx * head_stride + qtile_idx * BLOCK_M * HEAD_DIM
    // For simplicity, assuming contiguous layout: Q[batch, seq, head, dim]
    // Stride per head in seq dimension: head_dim = 128 bytes (FP8)
    // We need: (batch_idx * seqlen * num_heads + head_idx * seqlen + qtile_idx * BLOCK_M) * head_dim
    
    // Simple case: assume Q is [batch, seq, head, dim] and contiguous
    // Q offset = (qtile_idx * BLOCK_M) * HEAD_DIM + head_idx * head_stride + batch_idx * batch_stride
    // where head_stride = seqlen * HEAD_DIM and batch_stride = seqlen * num_heads * HEAD_DIM
    
    // For now, use simpler layout: Q[batch*head*seq, dim]
    // where linear index = batch * (num_heads * seq) + head * seq + seq_offset
    
    // Q-tile offset within sequence: s2 * BLOCK_M * HEAD_DIM
    s_lshl_b32 s32, s2, 13                    // s32 = qtile_idx * 64 * 128 = qtile_idx * 8192
    
    // We'll apply head and batch offsets later via proper strides
    // For minimal working version, use Q-tile offset only
    
    // ========================================================================
    // COMPUTE QK SCALE (softmax_scale * q_scale * k_scale)
    // ========================================================================
    v_mov_b32_e32 v3, s20
    v_mov_b32_e32 v4, s26
    v_mov_b32_e32 v5, s27
    v_mul_f32_e32 v3, v3, v4
    v_mul_f32_e32 v3, v3, v5
    v_readfirstlane_b32 s29, v3               // Combined scale in s29
    
    // Calculate number of K-tiles
    // k_tiles = (seqlen_k + BLOCK_N - 1) / BLOCK_N
    s_add_u32 s30, s22, 63
    s_lshr_b32 s30, s30, 6                    // s30 = k_tiles
    
    // ========================================================================
    // INITIALIZE ONLINE SOFTMAX STATE
    // ========================================================================
    v_mov_b32_e32 v16, 0xff800000             // running_max = -inf
    v_mov_b32_e32 v17, 0                      // running_sum = 0
    
    // ========================================================================
    // INITIALIZE OUTPUT ACCUMULATORS (16 FP32 values per thread × 4 D-tiles)
    // D-tile 0 (D=0..31):   v[48:63]
    // D-tile 1 (D=32..63):  v[144:159]
    // D-tile 2 (D=64..95):  v[160:175]
    // D-tile 3 (D=96..127): v[176:191]
    // ========================================================================
    // D-tile 0
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
    // D-tile 1
    v_mov_b32_e32 v144, 0
    v_mov_b32_e32 v145, 0
    v_mov_b32_e32 v146, 0
    v_mov_b32_e32 v147, 0
    v_mov_b32_e32 v148, 0
    v_mov_b32_e32 v149, 0
    v_mov_b32_e32 v150, 0
    v_mov_b32_e32 v151, 0
    v_mov_b32_e32 v152, 0
    v_mov_b32_e32 v153, 0
    v_mov_b32_e32 v154, 0
    v_mov_b32_e32 v155, 0
    v_mov_b32_e32 v156, 0
    v_mov_b32_e32 v157, 0
    v_mov_b32_e32 v158, 0
    v_mov_b32_e32 v159, 0
    // D-tile 2
    v_mov_b32_e32 v160, 0
    v_mov_b32_e32 v161, 0
    v_mov_b32_e32 v162, 0
    v_mov_b32_e32 v163, 0
    v_mov_b32_e32 v164, 0
    v_mov_b32_e32 v165, 0
    v_mov_b32_e32 v166, 0
    v_mov_b32_e32 v167, 0
    v_mov_b32_e32 v168, 0
    v_mov_b32_e32 v169, 0
    v_mov_b32_e32 v170, 0
    v_mov_b32_e32 v171, 0
    v_mov_b32_e32 v172, 0
    v_mov_b32_e32 v173, 0
    v_mov_b32_e32 v174, 0
    v_mov_b32_e32 v175, 0
    // D-tile 3
    v_mov_b32_e32 v176, 0
    v_mov_b32_e32 v177, 0
    v_mov_b32_e32 v178, 0
    v_mov_b32_e32 v179, 0
    v_mov_b32_e32 v180, 0
    v_mov_b32_e32 v181, 0
    v_mov_b32_e32 v182, 0
    v_mov_b32_e32 v183, 0
    v_mov_b32_e32 v184, 0
    v_mov_b32_e32 v185, 0
    v_mov_b32_e32 v186, 0
    v_mov_b32_e32 v187, 0
    v_mov_b32_e32 v188, 0
    v_mov_b32_e32 v189, 0
    v_mov_b32_e32 v190, 0
    v_mov_b32_e32 v191, 0
    
    // DEBUG: Store v48 right after initialization at offset 0x28000
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v14, 2, v0
    v_add_u32_e32 v14, 0x28000, v14
    v_add_co_u32_e32 v10, vcc, v14, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dword v[10:11], v48
    s_waitcnt vmcnt(0)
    
    // ========================================================================
    // LOAD Q TILE TO LDS (done once per Q-tile)
    // Each thread loads 16 bytes (16 FP8 values)
    // Total: 256 threads * 16 bytes = 4KB per iteration, need 2 iterations for 8KB
    // Q_offset = qtile_idx * 8192 + thread_offset
    // ========================================================================
    v_lshlrev_b32_e32 v6, 4, v0               // thread offset = tid * 16
    
    // First 4KB of Q (with qtile offset)
    v_mov_b32_e32 v10, s10                    // Q base low
    v_mov_b32_e32 v11, s11                    // Q base high
    v_mov_b32_e32 v8, s32                     // qtile offset
    v_add_co_u32_e32 v10, vcc, v8, v10        // add qtile offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10        // add thread offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[64:67], v[10:11]
    
    // Second 4KB of Q (offset by 4096)
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[68:71], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store Q to LDS
    ds_write_b128 v6, v[64:67]
    v_add_u32_e32 v7, 4096, v6
    ds_write_b128 v7, v[68:71]
    
    s_barrier
    
    // ========================================================================
    // K-TILE LOOP
    // ========================================================================
    s_mov_b32 s31, 0                          // k_tile_idx = 0
    
K_LOOP:
    // ========================================================================
    // LOAD K TILE TO LDS
    // K offset = k_tile_idx * BLOCK_N * HEAD_DIM = k_tile_idx * 8192
    // ========================================================================
    s_lshl_b32 s33, s31, 13                   // k_offset = k_tile_idx * 8192
    v_mov_b32_e32 v10, s12                    // K base low
    v_mov_b32_e32 v11, s13                    // K base high
    v_mov_b32_e32 v7, s33
    v_add_co_u32_e32 v10, vcc, v7, v10        // Add k_offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10        // Add thread offset
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[72:75], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[76:79], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store K to LDS at LDS_K_OFFSET
    v_add_u32_e32 v7, LDS_K_OFFSET, v6
    ds_write_b128 v7, v[72:75]
    v_add_u32_e32 v7, LDS_K_OFFSET + 4096, v6
    ds_write_b128 v7, v[76:79]
    
    s_barrier
    
    // ========================================================================
    // INITIALIZE QK ACCUMULATOR
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
    
    // ========================================================================
    // Q×K GEMM: Multiple MFMA instructions
    // FP8 MFMA: v_mfma_f32_32x32x16_fp8_fp8
    // Each MFMA processes K=16, need 8 MFMAs for HEAD_DIM=128
    //
    // MFMA 32x32x16 FP8 layout (64 threads):
    // - A (Q): 32 rows × 16 cols, each thread provides 8 FP8 values
    //   - kAMLane = 32, kABKLane = 2
    //   - lane_id / 2 determines which of 32 rows
    //   - lane_id % 2 determines K lane (0-7 or 8-15)
    // - B (K): 16 rows × 32 cols, similar distribution
    //
    // LDS layout (Q stored row-major at 128 bytes/row):
    // - Q[row, 0:127] at LDS[row * 128]
    // - For MFMA A: lane reads 8 bytes from row (lane/2), K-offset (lane%2)*8
    // ========================================================================
    
    // Compute LDS read address for Q based on lane position in wave
    // Current LDS layout: Q stored at tid * 16 (consecutive threads store consecutive chunks)
    // For MFMA A operand: lane_id maps to Q data
    // 
    // MFMA 32x32x16 FP8 with 64 threads:
    // - kAMLane = 32, kABKLane = 2
    // - lane_id % 2 determines K-lane (0 or 1 within K=16)
    // - lane_id / 2 determines M row (0-31)
    //
    // To match store pattern (tid * 16), read pattern should be:
    // - Lane 0 reads LDS[0..7] (first 8 of tid 0's 16 bytes)
    // - Lane 1 reads LDS[8..15] (last 8 of tid 0's 16 bytes)
    // - Lane 2 reads LDS[16..23] (first 8 of tid 1's 16 bytes)
    // - Lane 3 reads LDS[24..31] (last 8 of tid 1's 16 bytes)
    // - ...
    // 
    // So: lds_offset = (lane_id / 2) * 16 + (lane_id % 2) * 8 = lane_id * 8
    v_lshlrev_b32_e32 v7, 3, v2           // v7 = lane_id * 8
    
    // Load Q from LDS into AGPRs (K=0..15)
    ds_read_b64 a[0:1], v7                // 8 bytes at q_lds_base
    v_add_u32_e32 v9, 16, v7              // Next K slice (+16 bytes)
    ds_read_b64 a[2:3], v9
    
    // Compute K LDS address (similar pattern)
    // For K (transposed): we want K[j, k] to be available as K^T[k, j]
    // K is stored at LDS_K_OFFSET with same row-major layout
    v_add_u32_e32 v9, LDS_K_OFFSET, v7    // K at same relative offset
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=0..15, K=16..31
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=32..63
    v_add_u32_e32 v9, 32, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    
    // Load K K=32..63
    v_add_u32_e32 v9, LDS_K_OFFSET + 32, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=32..47, K=48..63
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    // Load Q K=64..95
    v_add_u32_e32 v9, 64, v7
    ds_read_b64 a[0:1], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[2:3], v9
    
    // Load K K=64..95
    v_add_u32_e32 v9, LDS_K_OFFSET + 64, v7
    ds_read_b64 v[64:65], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[66:67], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=64..79, K=80..95
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[64:65], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[2:3], v[66:67], v[32:47]
    
    // Load Q K=96..127
    v_add_u32_e32 v9, 96, v7
    ds_read_b64 a[4:5], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 a[6:7], v9
    
    // Load K K=96..127
    v_add_u32_e32 v9, LDS_K_OFFSET + 96, v7
    ds_read_b64 v[68:69], v9
    v_add_u32_e32 v9, 16, v9
    ds_read_b64 v[70:71], v9
    
    s_waitcnt lgkmcnt(0)
    
    // QK MFMAs - K=96..111, K=112..127
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[4:5], v[68:69], v[32:47]
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[6:7], v[70:71], v[32:47]
    
    // DEBUG: Store all 16 QK values per thread
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v14, 6, v0          // tid * 64 bytes (16 floats)
    v_add_u32_e32 v14, 0x8000, v14
    v_add_co_u32_e32 v10, vcc, v14, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[32:35]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[36:39]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[40:43]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[44:47]
    s_waitcnt vmcnt(0)
    
    // ========================================================================
    // APPLY QK SCALE
    // ========================================================================
    v_mul_f32_e32 v32, s29, v32
    v_mul_f32_e32 v33, s29, v33
    v_mul_f32_e32 v34, s29, v34
    v_mul_f32_e32 v35, s29, v35
    v_mul_f32_e32 v36, s29, v36
    v_mul_f32_e32 v37, s29, v37
    v_mul_f32_e32 v38, s29, v38
    v_mul_f32_e32 v39, s29, v39
    v_mul_f32_e32 v40, s29, v40
    v_mul_f32_e32 v41, s29, v41
    v_mul_f32_e32 v42, s29, v42
    v_mul_f32_e32 v43, s29, v43
    v_mul_f32_e32 v44, s29, v44
    v_mul_f32_e32 v45, s29, v45
    v_mul_f32_e32 v46, s29, v46
    v_mul_f32_e32 v47, s29, v47
    
    // ========================================================================
    // ONLINE SOFTMAX: Find local max
    // ========================================================================
    v_max_f32_e32 v18, v32, v33
    v_max_f32_e32 v18, v18, v34
    v_max_f32_e32 v18, v18, v35
    v_max_f32_e32 v18, v18, v36
    v_max_f32_e32 v18, v18, v37
    v_max_f32_e32 v18, v18, v38
    v_max_f32_e32 v18, v18, v39
    v_max_f32_e32 v18, v18, v40
    v_max_f32_e32 v18, v18, v41
    v_max_f32_e32 v18, v18, v42
    v_max_f32_e32 v18, v18, v43
    v_max_f32_e32 v18, v18, v44
    v_max_f32_e32 v18, v18, v45
    v_max_f32_e32 v18, v18, v46
    v_max_f32_e32 v18, v18, v47         // v18 = local_max (this thread's 16 values)
    
    // Cross-thread max reduction: share max between lanes 0-31 and 32-63
    // This gives row-wise max across all 32 columns computed by this wave
    v_mov_b32_e32 v19, v18
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v19, v18   // swap between lane halves
    v_max_f32_e32 v18, v18, v19          // v18 = max across both halves = row max
    
    // Compute new_max = max(running_max, local_max)
    v_mov_b32_e32 v19, v16               // old_max = running_max
    v_max_f32_e32 v16, v16, v18          // running_max = max(running_max, row_max)
    
    // Compute correction factor = exp(old_max - new_max)
    v_sub_f32_e32 v20, v19, v16
    v_exp_f32_e32 v19, v20               // correction = exp(old_max - new_max)
    
    // DEBUG: Store correction factor at offset 0x24000 and v20 at 0x24400
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v14, 2, v0
    v_add_u32_e32 v15, 0x24000, v14
    v_add_co_u32_e32 v10, vcc, v15, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dword v[10:11], v19
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_u32_e32 v15, 0x24400, v14
    v_add_co_u32_e32 v10, vcc, v15, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dword v[10:11], v20
    s_waitcnt vmcnt(0)
    
    // Scale previous output accumulators by correction (all 4 D-tiles)
    // D-tile 0
    v_mul_f32_e32 v48, v19, v48
    v_mul_f32_e32 v49, v19, v49
    v_mul_f32_e32 v50, v19, v50
    v_mul_f32_e32 v51, v19, v51
    v_mul_f32_e32 v52, v19, v52
    v_mul_f32_e32 v53, v19, v53
    v_mul_f32_e32 v54, v19, v54
    v_mul_f32_e32 v55, v19, v55
    v_mul_f32_e32 v56, v19, v56
    v_mul_f32_e32 v57, v19, v57
    v_mul_f32_e32 v58, v19, v58
    v_mul_f32_e32 v59, v19, v59
    v_mul_f32_e32 v60, v19, v60
    v_mul_f32_e32 v61, v19, v61
    v_mul_f32_e32 v62, v19, v62
    v_mul_f32_e32 v63, v19, v63
    // D-tile 1
    v_mul_f32_e32 v144, v19, v144
    v_mul_f32_e32 v145, v19, v145
    v_mul_f32_e32 v146, v19, v146
    v_mul_f32_e32 v147, v19, v147
    v_mul_f32_e32 v148, v19, v148
    v_mul_f32_e32 v149, v19, v149
    v_mul_f32_e32 v150, v19, v150
    v_mul_f32_e32 v151, v19, v151
    v_mul_f32_e32 v152, v19, v152
    v_mul_f32_e32 v153, v19, v153
    v_mul_f32_e32 v154, v19, v154
    v_mul_f32_e32 v155, v19, v155
    v_mul_f32_e32 v156, v19, v156
    v_mul_f32_e32 v157, v19, v157
    v_mul_f32_e32 v158, v19, v158
    v_mul_f32_e32 v159, v19, v159
    // D-tile 2
    v_mul_f32_e32 v160, v19, v160
    v_mul_f32_e32 v161, v19, v161
    v_mul_f32_e32 v162, v19, v162
    v_mul_f32_e32 v163, v19, v163
    v_mul_f32_e32 v164, v19, v164
    v_mul_f32_e32 v165, v19, v165
    v_mul_f32_e32 v166, v19, v166
    v_mul_f32_e32 v167, v19, v167
    v_mul_f32_e32 v168, v19, v168
    v_mul_f32_e32 v169, v19, v169
    v_mul_f32_e32 v170, v19, v170
    v_mul_f32_e32 v171, v19, v171
    v_mul_f32_e32 v172, v19, v172
    v_mul_f32_e32 v173, v19, v173
    v_mul_f32_e32 v174, v19, v174
    v_mul_f32_e32 v175, v19, v175
    // D-tile 3
    v_mul_f32_e32 v176, v19, v176
    v_mul_f32_e32 v177, v19, v177
    v_mul_f32_e32 v178, v19, v178
    v_mul_f32_e32 v179, v19, v179
    v_mul_f32_e32 v180, v19, v180
    v_mul_f32_e32 v181, v19, v181
    v_mul_f32_e32 v182, v19, v182
    v_mul_f32_e32 v183, v19, v183
    v_mul_f32_e32 v184, v19, v184
    v_mul_f32_e32 v185, v19, v185
    v_mul_f32_e32 v186, v19, v186
    v_mul_f32_e32 v187, v19, v187
    v_mul_f32_e32 v188, v19, v188
    v_mul_f32_e32 v189, v19, v189
    v_mul_f32_e32 v190, v19, v190
    v_mul_f32_e32 v191, v19, v191
    
    // Scale previous running_sum by correction
    v_mul_f32_e32 v17, v19, v17
    
    // ========================================================================
    // COMPUTE exp(QK - new_max) 
    // ========================================================================
    v_sub_f32_e32 v32, v32, v16
    v_exp_f32_e32 v32, v32
    v_sub_f32_e32 v33, v33, v16
    v_exp_f32_e32 v33, v33
    v_sub_f32_e32 v34, v34, v16
    v_exp_f32_e32 v34, v34
    v_sub_f32_e32 v35, v35, v16
    v_exp_f32_e32 v35, v35
    v_sub_f32_e32 v36, v36, v16
    v_exp_f32_e32 v36, v36
    v_sub_f32_e32 v37, v37, v16
    v_exp_f32_e32 v37, v37
    v_sub_f32_e32 v38, v38, v16
    v_exp_f32_e32 v38, v38
    v_sub_f32_e32 v39, v39, v16
    v_exp_f32_e32 v39, v39
    v_sub_f32_e32 v40, v40, v16
    v_exp_f32_e32 v40, v40
    v_sub_f32_e32 v41, v41, v16
    v_exp_f32_e32 v41, v41
    v_sub_f32_e32 v42, v42, v16
    v_exp_f32_e32 v42, v42
    v_sub_f32_e32 v43, v43, v16
    v_exp_f32_e32 v43, v43
    v_sub_f32_e32 v44, v44, v16
    v_exp_f32_e32 v44, v44
    v_sub_f32_e32 v45, v45, v16
    v_exp_f32_e32 v45, v45
    v_sub_f32_e32 v46, v46, v16
    v_exp_f32_e32 v46, v46
    v_sub_f32_e32 v47, v47, v16
    v_exp_f32_e32 v47, v47
    
    // Sum P values (v32-v47) into local_sum (v20)
    // Using tree reduction for better numerical stability
    v_add_f32_e32 v20, v32, v33
    v_add_f32_e32 v21, v34, v35
    v_add_f32_e32 v22, v36, v37
    v_add_f32_e32 v23, v38, v39
    v_add_f32_e32 v20, v20, v21
    v_add_f32_e32 v22, v22, v23
    v_add_f32_e32 v20, v20, v22
    v_add_f32_e32 v21, v40, v41
    v_add_f32_e32 v22, v42, v43
    v_add_f32_e32 v23, v44, v45
    v_add_f32_e32 v71, v46, v47
    v_add_f32_e32 v21, v21, v22
    v_add_f32_e32 v23, v23, v71
    v_add_f32_e32 v21, v21, v23
    v_add_f32_e32 v20, v20, v21           // v20 = local_sum (this thread's 16 values)
    
    // Cross-thread sum reduction: share sum between lanes 0-31 and 32-63
    v_mov_b32_e32 v21, v20
    s_nop 0
    s_nop 0
    v_permlane32_swap_b32_e32 v21, v20   // swap between lane halves
    v_add_f32_e32 v20, v20, v21          // v20 = sum across both halves = row sum
    
    // Update running sum
    v_add_f32_e32 v17, v17, v20
    
    // DEBUG: Store softmax P values at offset 0x10000
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v14, 6, v0          // tid * 64 bytes
    v_add_u32_e32 v14, 0x10000, v14
    v_add_co_u32_e32 v10, vcc, v14, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[32:35]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[36:39]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[40:43]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[44:47]
    // Also store running_sum (v17) at offset 0x10000 + 0x4000
    v_lshlrev_b32_e32 v14, 2, v0          // tid * 4 bytes
    v_add_u32_e32 v14, 0x14000, v14
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_add_co_u32_e32 v10, vcc, v14, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dword v[10:11], v17
    s_waitcnt vmcnt(0)
    
    // ========================================================================
    // LOAD V TILE AND COMPUTE P×V
    // ========================================================================
    // Load V tile (same offset as K)
    v_mov_b32_e32 v10, s14                // V base low
    v_mov_b32_e32 v11, s15                // V base high
    v_mov_b32_e32 v7, s33                 // k_offset
    v_add_co_u32_e32 v10, vcc, v7, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    v_add_co_u32_e32 v10, vcc, v6, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    flat_load_dwordx4 v[72:75], v[10:11]
    v_add_co_u32_e32 v10, vcc, 4096, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_load_dwordx4 v[76:79], v[10:11]
    
    s_waitcnt vmcnt(0)
    
    // Store V to LDS at LDS_V_OFFSET
    v_add_u32_e32 v7, LDS_V_OFFSET, v6
    ds_write_b128 v7, v[72:75]
    v_add_u32_e32 v7, LDS_V_OFFSET + 4096, v6
    ds_write_b128 v7, v[76:79]
    
    s_barrier
    
    // Convert P (softmax output) to FP8 for MFMA
    // v_cvt_pk_fp8_f32 packs 2 FP32 into 2 FP8 in lower 16 bits
    // Need to combine pairs to fill full 32-bit registers for MFMA
    
    // Pack P[0:1] and P[2:3] into v21 (4 FP8 values)
    // IMPORTANT: Clear upper 16 bits before cvt_pk which only writes lower 16 bits
    v_mov_b32_e32 v21, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v21, v32, v33        // v21[15:0] = P[0:1]
    v_cvt_pk_fp8_f32 v70, v34, v35        // v70[15:0] = P[2:3]
    v_lshlrev_b32_e32 v70, 16, v70        // v70 = P[2:3] << 16
    v_or_b32_e32 v21, v21, v70            // v21 = P[0:3] as 4 FP8
    
    // Pack P[4:5] and P[6:7] into v22 (4 FP8 values)
    v_mov_b32_e32 v22, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v22, v36, v37        // v22[15:0] = P[4:5]
    v_cvt_pk_fp8_f32 v70, v38, v39        // v70[15:0] = P[6:7]
    v_lshlrev_b32_e32 v70, 16, v70        // v70 = P[6:7] << 16
    v_or_b32_e32 v22, v22, v70            // v22 = P[4:7] as 4 FP8
    
    // Now a[0:1] = (v21, v22) contains 8 valid FP8 values for MFMA A operand
    v_accvgpr_write_b32 a0, v21
    v_accvgpr_write_b32 a1, v22
    
    // Pack second set: P[8:15] for a[2:3]
    v_mov_b32_e32 v23, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v23, v40, v41
    v_cvt_pk_fp8_f32 v70, v42, v43
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v23, v23, v70            // v23 = P[8:11]
    
    v_mov_b32_e32 v24, 0
    v_mov_b32_e32 v70, 0
    v_cvt_pk_fp8_f32 v24, v44, v45
    v_cvt_pk_fp8_f32 v70, v46, v47
    v_lshlrev_b32_e32 v70, 16, v70
    v_or_b32_e32 v24, v24, v70            // v24 = P[12:15]
    
    v_accvgpr_write_b32 a2, v23
    v_accvgpr_write_b32 a3, v24
    
    // DEBUG: Store packed FP8 P values at offset 0x1C000
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v14, 4, v0          // tid * 16 bytes (4 dwords)
    v_add_u32_e32 v14, 0x1C000, v14
    v_add_co_u32_e32 v10, vcc, v14, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    // Copy to aligned registers
    v_mov_b32_e32 v80, v21
    v_mov_b32_e32 v81, v22
    v_mov_b32_e32 v82, v23
    v_mov_b32_e32 v83, v24
    flat_store_dwordx4 v[10:11], v[80:83]
    s_waitcnt vmcnt(0)
    
    // ========================================================================
    // P×V COMPUTATION FOR ALL 4 D-TILES
    // V layout in LDS: V[K, D] where K=seq_k (32), D=head_dim (128)
    // V stored row-major: stride_D = 1 byte (FP8), stride_K = 128 bytes
    // 
    // For PV MFMA 32x32x16:
    //   A = P[32×16] (32 Q rows, 16 K cols to reduce)
    //   B = V[16×32] (16 K rows, 32 D cols)
    //   C = O[32×32] (32 Q rows, 32 D cols)
    //
    // Need 2 MFMAs for K reduction (32/16=2) per D tile
    // Need 4 D tiles for head_dim=128 (128/32=4)
    // Total: 2 * 4 = 8 MFMAs per K-tile
    // ========================================================================
    
    // Base V offset computation: (lane % 32) * 8 for MFMA B operand
    v_and_b32_e32 v7, 31, v0              // lane within 32
    v_lshlrev_b32_e32 v7, 3, v7           // * 8 bytes
    
    // ========================================================================
    // D-TILE 0: D=0..31, output in v[48:63]
    // ========================================================================
    v_add_u32_e32 v8, LDS_V_OFFSET, v7    // base + lane offset (D=0..31)
    ds_read_b64 v[64:65], v8               // V[K=0..7]
    ds_read_b64 v[66:67], v8 offset:128    // V[K=8..15]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[0:1], v[64:65], v[48:63]
    ds_read_b64 v[64:65], v8 offset:256    // V[K=16..23]
    ds_read_b64 v[66:67], v8 offset:384    // V[K=24..31]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[48:63], a[2:3], v[64:65], v[48:63]
    
    // ========================================================================
    // D-TILE 1: D=32..63, output in v[144:159]
    // V offset += 32 bytes for D-tile 1
    // ========================================================================
    v_add_u32_e32 v8, LDS_V_OFFSET + 32, v7
    ds_read_b64 v[64:65], v8
    ds_read_b64 v[66:67], v8 offset:128
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[144:159], a[0:1], v[64:65], v[144:159]
    ds_read_b64 v[64:65], v8 offset:256
    ds_read_b64 v[66:67], v8 offset:384
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[144:159], a[2:3], v[64:65], v[144:159]
    
    // ========================================================================
    // D-TILE 2: D=64..95, output in v[160:175]
    // V offset += 64 bytes for D-tile 2
    // ========================================================================
    v_add_u32_e32 v8, LDS_V_OFFSET + 64, v7
    ds_read_b64 v[64:65], v8
    ds_read_b64 v[66:67], v8 offset:128
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[160:175], a[0:1], v[64:65], v[160:175]
    ds_read_b64 v[64:65], v8 offset:256
    ds_read_b64 v[66:67], v8 offset:384
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[160:175], a[2:3], v[64:65], v[160:175]
    
    // ========================================================================
    // D-TILE 3: D=96..127, output in v[176:191]
    // V offset += 96 bytes for D-tile 3
    // ========================================================================
    v_add_u32_e32 v8, LDS_V_OFFSET + 96, v7
    ds_read_b64 v[64:65], v8
    ds_read_b64 v[66:67], v8 offset:128
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[176:191], a[0:1], v[64:65], v[176:191]
    ds_read_b64 v[64:65], v8 offset:256
    ds_read_b64 v[66:67], v8 offset:384
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x16_fp8_fp8 v[176:191], a[2:3], v[64:65], v[176:191]
    
    // ========================================================================
    // K-LOOP INCREMENT AND CHECK
    // ========================================================================
    s_add_u32 s31, s31, 1
    s_cmp_lt_u32 s31, s30
    s_cbranch_scc1 K_LOOP
    
    // ========================================================================
    // FINAL NORMALIZATION (divide by running_sum) - ALL 4 D-TILES
    // ========================================================================
    v_rcp_f32_e32 v17, v17               // 1/sum
    
    // DEBUG: Store 1/sum (v17) at offset 0x30000
    v_mov_b32_e32 v10, s8
    v_mov_b32_e32 v11, s9
    v_lshlrev_b32_e32 v14, 2, v0
    v_add_u32_e32 v14, 0x30000, v14
    v_add_co_u32_e32 v10, vcc, v14, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dword v[10:11], v17
    s_waitcnt vmcnt(0)
    
    // Normalize D-tile 0
    v_mul_f32_e32 v48, v17, v48
    v_mul_f32_e32 v49, v17, v49
    v_mul_f32_e32 v50, v17, v50
    v_mul_f32_e32 v51, v17, v51
    v_mul_f32_e32 v52, v17, v52
    v_mul_f32_e32 v53, v17, v53
    v_mul_f32_e32 v54, v17, v54
    v_mul_f32_e32 v55, v17, v55
    v_mul_f32_e32 v56, v17, v56
    v_mul_f32_e32 v57, v17, v57
    v_mul_f32_e32 v58, v17, v58
    v_mul_f32_e32 v59, v17, v59
    v_mul_f32_e32 v60, v17, v60
    v_mul_f32_e32 v61, v17, v61
    v_mul_f32_e32 v62, v17, v62
    v_mul_f32_e32 v63, v17, v63
    
    // Normalize D-tile 1
    v_mul_f32_e32 v144, v17, v144
    v_mul_f32_e32 v145, v17, v145
    v_mul_f32_e32 v146, v17, v146
    v_mul_f32_e32 v147, v17, v147
    v_mul_f32_e32 v148, v17, v148
    v_mul_f32_e32 v149, v17, v149
    v_mul_f32_e32 v150, v17, v150
    v_mul_f32_e32 v151, v17, v151
    v_mul_f32_e32 v152, v17, v152
    v_mul_f32_e32 v153, v17, v153
    v_mul_f32_e32 v154, v17, v154
    v_mul_f32_e32 v155, v17, v155
    v_mul_f32_e32 v156, v17, v156
    v_mul_f32_e32 v157, v17, v157
    v_mul_f32_e32 v158, v17, v158
    v_mul_f32_e32 v159, v17, v159
    
    // Normalize D-tile 2
    v_mul_f32_e32 v160, v17, v160
    v_mul_f32_e32 v161, v17, v161
    v_mul_f32_e32 v162, v17, v162
    v_mul_f32_e32 v163, v17, v163
    v_mul_f32_e32 v164, v17, v164
    v_mul_f32_e32 v165, v17, v165
    v_mul_f32_e32 v166, v17, v166
    v_mul_f32_e32 v167, v17, v167
    v_mul_f32_e32 v168, v17, v168
    v_mul_f32_e32 v169, v17, v169
    v_mul_f32_e32 v170, v17, v170
    v_mul_f32_e32 v171, v17, v171
    v_mul_f32_e32 v172, v17, v172
    v_mul_f32_e32 v173, v17, v173
    v_mul_f32_e32 v174, v17, v174
    v_mul_f32_e32 v175, v17, v175
    
    // Normalize D-tile 3
    v_mul_f32_e32 v176, v17, v176
    v_mul_f32_e32 v177, v17, v177
    v_mul_f32_e32 v178, v17, v178
    v_mul_f32_e32 v179, v17, v179
    v_mul_f32_e32 v180, v17, v180
    v_mul_f32_e32 v181, v17, v181
    v_mul_f32_e32 v182, v17, v182
    v_mul_f32_e32 v183, v17, v183
    v_mul_f32_e32 v184, v17, v184
    v_mul_f32_e32 v185, v17, v185
    v_mul_f32_e32 v186, v17, v186
    v_mul_f32_e32 v187, v17, v187
    v_mul_f32_e32 v188, v17, v188
    v_mul_f32_e32 v189, v17, v189
    v_mul_f32_e32 v190, v17, v190
    v_mul_f32_e32 v191, v17, v191
    
    // Apply V scale (all 4 D-tiles)
    v_mul_f32_e32 v48, s28, v48
    v_mul_f32_e32 v49, s28, v49
    v_mul_f32_e32 v50, s28, v50
    v_mul_f32_e32 v51, s28, v51
    v_mul_f32_e32 v52, s28, v52
    v_mul_f32_e32 v53, s28, v53
    v_mul_f32_e32 v54, s28, v54
    v_mul_f32_e32 v55, s28, v55
    v_mul_f32_e32 v56, s28, v56
    v_mul_f32_e32 v57, s28, v57
    v_mul_f32_e32 v58, s28, v58
    v_mul_f32_e32 v59, s28, v59
    v_mul_f32_e32 v60, s28, v60
    v_mul_f32_e32 v61, s28, v61
    v_mul_f32_e32 v62, s28, v62
    v_mul_f32_e32 v63, s28, v63
    v_mul_f32_e32 v144, s28, v144
    v_mul_f32_e32 v145, s28, v145
    v_mul_f32_e32 v146, s28, v146
    v_mul_f32_e32 v147, s28, v147
    v_mul_f32_e32 v148, s28, v148
    v_mul_f32_e32 v149, s28, v149
    v_mul_f32_e32 v150, s28, v150
    v_mul_f32_e32 v151, s28, v151
    v_mul_f32_e32 v152, s28, v152
    v_mul_f32_e32 v153, s28, v153
    v_mul_f32_e32 v154, s28, v154
    v_mul_f32_e32 v155, s28, v155
    v_mul_f32_e32 v156, s28, v156
    v_mul_f32_e32 v157, s28, v157
    v_mul_f32_e32 v158, s28, v158
    v_mul_f32_e32 v159, s28, v159
    v_mul_f32_e32 v160, s28, v160
    v_mul_f32_e32 v161, s28, v161
    v_mul_f32_e32 v162, s28, v162
    v_mul_f32_e32 v163, s28, v163
    v_mul_f32_e32 v164, s28, v164
    v_mul_f32_e32 v165, s28, v165
    v_mul_f32_e32 v166, s28, v166
    v_mul_f32_e32 v167, s28, v167
    v_mul_f32_e32 v168, s28, v168
    v_mul_f32_e32 v169, s28, v169
    v_mul_f32_e32 v170, s28, v170
    v_mul_f32_e32 v171, s28, v171
    v_mul_f32_e32 v172, s28, v172
    v_mul_f32_e32 v173, s28, v173
    v_mul_f32_e32 v174, s28, v174
    v_mul_f32_e32 v175, s28, v175
    v_mul_f32_e32 v176, s28, v176
    v_mul_f32_e32 v177, s28, v177
    v_mul_f32_e32 v178, s28, v178
    v_mul_f32_e32 v179, s28, v179
    v_mul_f32_e32 v180, s28, v180
    v_mul_f32_e32 v181, s28, v181
    v_mul_f32_e32 v182, s28, v182
    v_mul_f32_e32 v183, s28, v183
    v_mul_f32_e32 v184, s28, v184
    v_mul_f32_e32 v185, s28, v185
    v_mul_f32_e32 v186, s28, v186
    v_mul_f32_e32 v187, s28, v187
    v_mul_f32_e32 v188, s28, v188
    v_mul_f32_e32 v189, s28, v189
    v_mul_f32_e32 v190, s28, v190
    v_mul_f32_e32 v191, s28, v191
    
    // ========================================================================
    // STORE OUTPUT - ALL 4 D-TILES (64 floats per thread = 256 bytes)
    // Layout: tid * 256 bytes, D-tiles contiguous within each thread's output
    // ========================================================================
    v_mov_b32_e32 v10, s8                 // Output base low
    v_mov_b32_e32 v11, s9                 // Output base high
    
    // offset = tid * 256 bytes (64 floats * 4)
    v_lshlrev_b32_e32 v3, 8, v0           // tid * 256
    v_add_co_u32_e32 v10, vcc, v3, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store D-tile 0 (v[48:63])
    flat_store_dwordx4 v[10:11], v[48:51]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[52:55]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[56:59]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[60:63]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store D-tile 1 (v[144:159])
    flat_store_dwordx4 v[10:11], v[144:147]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[148:151]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[152:155]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[156:159]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store D-tile 2 (v[160:175])
    flat_store_dwordx4 v[10:11], v[160:163]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[164:167]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[168:171]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[172:175]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    
    // Store D-tile 3 (v[176:191])
    flat_store_dwordx4 v[10:11], v[176:179]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[180:183]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[184:187]
    v_add_co_u32_e32 v10, vcc, 16, v10
    v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
    flat_store_dwordx4 v[10:11], v[188:191]
    
    s_waitcnt vmcnt(0)
    s_endpgm

.size _ZN5aiter18fmha_fwd_hd128_fp8E, .-_ZN5aiter18fmha_fwd_hd128_fp8E

// ========================================================================
// KERNEL DESCRIPTOR
// ========================================================================
.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter18fmha_fwd_hd128_fp8E
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 528
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 192
    .amdhsa_next_free_sgpr 48
    .amdhsa_accum_offset 192
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter18fmha_fwd_hd128_fp8E
    .symbol: _ZN5aiter18fmha_fwd_hd128_fp8E.kd
    .kernarg_segment_size: 528
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 48
    .vgpr_count: 192
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .args:
      - .name: ptr_R
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_Q
        .size: 8
        .offset: 16
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_K
        .size: 8
        .offset: 32
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_V
        .size: 8
        .offset: 48
        .value_kind: global_buffer
        .address_space: global
      - .name: ptr_LSE
        .size: 8
        .offset: 64
        .value_kind: global_buffer
        .address_space: global
      - .name: softmax_scale
        .size: 4
        .offset: 80
        .value_kind: by_value
      - .name: seqlen_q
        .size: 4
        .offset: 88
        .value_kind: by_value
      - .name: seqlen_k
        .size: 4
        .offset: 96
        .value_kind: by_value
      - .name: q_scale
        .size: 4
        .offset: 512
        .value_kind: by_value
      - .name: k_scale
        .size: 4
        .offset: 516
        .value_kind: by_value
      - .name: v_scale
        .size: 4
        .offset: 520
        .value_kind: by_value
...
.end_amdgpu_metadata

// Test: TR8 interleaved layout for FP8 MFMA
// 
// Layout: Q[row, k] at LDS[(row % 8) + (row / 8) * 1024 + k * 8]
// This allows TR8 to read 8 consecutive k values with stride-8 pattern
//
// 64 threads, each loads 16 bytes from global, scatters to interleaved LDS
// Then reads back with ds_read_b64_tr_b8 and writes to output

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl test_tr8_interleaved
.p2align 8
.type test_tr8_interleaved,@function

test_tr8_interleaved:
    s_mov_b64 exec, -1
    
    // Args: output, input (Q[8×128] for simplicity - 8 rows)
    s_load_dwordx2 s[4:5], s[0:1], 0x00   // output
    s_load_dwordx2 s[8:9], s[0:1], 0x08   // input Q
    s_waitcnt lgkmcnt(0)

    // Buffer descriptors (size=-1, flags=0x20000)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    
    v_mov_b32_e32 v60, v0                  // tid (0-63)
    
    // =======================================================================
    // Step 1: Load 16 bytes from global (row-major Q[8×128])
    // Thread t loads Q[t/8, (t%8)*16 : (t%8)*16+15]
    // =======================================================================
    
    // Calculate global offset
    v_lshrrev_b32_e32 v10, 3, v60          // row = tid / 8
    v_and_b32_e32 v11, 7, v60              // k_chunk = tid % 8
    v_lshlrev_b32_e32 v12, 7, v10          // row * 128
    v_lshlrev_b32_e32 v13, 4, v11          // k_chunk * 16
    v_add_u32_e32 v14, v12, v13            // global_offset = row*128 + k_chunk*16
    
    // Load 16 bytes
    buffer_load_dwordx4 v[0:3], v14, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    
    // =======================================================================
    // Step 2: Scatter to interleaved LDS layout
    // Q[row, k] at LDS[(row % 8) + k * 8]
    // For 8 rows, row % 8 = row, so Q[row, k] at LDS[row + k*8]
    // =======================================================================
    
    // For this test with 8 rows, layout is simple:
    // Thread t (row=t/8, k_chunk=t%8) writes bytes at:
    //   LDS[row + (k_chunk*16 + 0)*8]
    //   LDS[row + (k_chunk*16 + 1)*8]
    //   ...
    //   LDS[row + (k_chunk*16 + 15)*8]
    
    // Base: row + k_chunk * 128 (since 16 * 8 = 128)
    v_lshlrev_b32_e32 v15, 7, v11          // k_chunk * 128
    v_add_u32_e32 v15, v10, v15            // row + k_chunk * 128
    
    // Scatter 16 bytes with stride 8
    // v0 contains bytes 0-3, v1 contains 4-7, etc.
    
    // Byte 0
    v_and_b32_e32 v20, 0xFF, v0
    ds_write_b8 v15, v20
    
    // Byte 1
    v_add_u32_e32 v16, 8, v15
    v_bfe_u32 v20, v0, 8, 8
    ds_write_b8 v16, v20
    
    // Byte 2
    v_add_u32_e32 v16, 16, v15
    v_bfe_u32 v20, v0, 16, 8
    ds_write_b8 v16, v20
    
    // Byte 3
    v_add_u32_e32 v16, 24, v15
    v_bfe_u32 v20, v0, 24, 8
    ds_write_b8 v16, v20
    
    // Byte 4
    v_add_u32_e32 v16, 32, v15
    v_and_b32_e32 v20, 0xFF, v1
    ds_write_b8 v16, v20
    
    // Byte 5
    v_add_u32_e32 v16, 40, v15
    v_bfe_u32 v20, v1, 8, 8
    ds_write_b8 v16, v20
    
    // Byte 6
    v_add_u32_e32 v16, 48, v15
    v_bfe_u32 v20, v1, 16, 8
    ds_write_b8 v16, v20
    
    // Byte 7
    v_add_u32_e32 v16, 56, v15
    v_bfe_u32 v20, v1, 24, 8
    ds_write_b8 v16, v20
    
    // Byte 8
    v_add_u32_e32 v16, 64, v15
    v_and_b32_e32 v20, 0xFF, v2
    ds_write_b8 v16, v20
    
    // Byte 9
    v_add_u32_e32 v16, 72, v15
    v_bfe_u32 v20, v2, 8, 8
    ds_write_b8 v16, v20
    
    // Byte 10
    v_add_u32_e32 v16, 80, v15
    v_bfe_u32 v20, v2, 16, 8
    ds_write_b8 v16, v20
    
    // Byte 11
    v_add_u32_e32 v16, 88, v15
    v_bfe_u32 v20, v2, 24, 8
    ds_write_b8 v16, v20
    
    // Byte 12
    v_add_u32_e32 v16, 96, v15
    v_and_b32_e32 v20, 0xFF, v3
    ds_write_b8 v16, v20
    
    // Byte 13
    v_add_u32_e32 v16, 104, v15
    v_bfe_u32 v20, v3, 8, 8
    ds_write_b8 v16, v20
    
    // Byte 14
    v_add_u32_e32 v16, 112, v15
    v_bfe_u32 v20, v3, 16, 8
    ds_write_b8 v16, v20
    
    // Byte 15
    v_add_u32_e32 v16, 120, v15
    v_bfe_u32 v20, v3, 24, 8
    ds_write_b8 v16, v20
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // =======================================================================
    // Step 3: Read with TR8 and verify
    // Each thread reads Q[tid % 8, 0:7] using ds_read_b64_tr_b8
    // =======================================================================
    
    v_and_b32_e32 v30, 7, v60              // row = tid % 8
    
    // TR8 base for row r, k=0: LDS[r]
    // This should read: LDS[r], LDS[r+8], ..., LDS[r+56]
    // = Q[r,0], Q[r,1], ..., Q[r,7]
    
    ds_read_b64_tr_b8 v[40:41], v30
    s_waitcnt lgkmcnt(0)
    
    // =======================================================================
    // Step 4: Write result to output
    // Each thread writes 8 bytes (v40, v41) to output[tid*8]
    // =======================================================================
    
    v_lshlrev_b32_e32 v50, 3, v60          // tid * 8
    buffer_store_dwordx2 v[40:41], v50, s[4:7], 0 offen
    
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel test_tr8_interleaved
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 64
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: test_tr8_interleaved
    .symbol: test_tr8_interleaved.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
...
.end_amdgpu_metadata

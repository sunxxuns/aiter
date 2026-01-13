// Raw MFMA throughput test - no memory, just MFMAs
// Measure peak MFMA TF/s

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter13bench_mfma_rawE
.p2align 8
.type _ZN5aiter13bench_mfma_rawE,@function

_ZN5aiter13bench_mfma_rawE:
    s_mov_b64 exec, -1
    
    // Just clear accumulators and run many MFMAs
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
    
    // Dummy operands (doesn't matter for throughput test)
    v_mov_b32_e32 v30, 0x38383838  // 4x FP8 1.0
    v_mov_b32_e32 v31, 0x38383838
    v_mov_b32_e32 v32, 0x38383838
    v_mov_b32_e32 v33, 0x38383838
    
    // 256 MFMAs to measure throughput
    .rept 256
        v_mfma_f32_32x32x16_fp8_fp8 v[0:15], v[30:31], v[32:33], v[0:15]
    .endr
    
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter13bench_mfma_rawE
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 48
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 48
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter13bench_mfma_rawE
    .symbol: _ZN5aiter13bench_mfma_rawE.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 48
    .max_flat_workgroup_size: 64
...
.end_amdgpu_metadata

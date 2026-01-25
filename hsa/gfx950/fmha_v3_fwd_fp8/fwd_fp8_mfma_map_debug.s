.amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"

.text
.globl _fwd_fp8_mfma_map_debug
.p2align 8
.type _fwd_fp8_mfma_map_debug,@function

// Debug kernel to verify MFMA K=64 mapping.
// Args:
//  s[0:1] = kernarg ptr
//  arg0: out_ptr (float32, size = 64 lanes * 16 floats)
//  arg1: in_ptr (uint32 packed A/B regs, 64 lanes * 16 dwords)
_fwd_fp8_mfma_map_debug:
    s_mov_b64 exec, -1

    // Load kernel args
    s_load_dwordx2 s[4:5], s[0:1], 0    // out_ptr
    s_load_dwordx2 s[8:9], s[0:1], 8    // in_ptr
    s_waitcnt lgkmcnt(0)

    // Lane id (0..63)
    v_mov_b32_e32 v30, v0
    v_and_b32_e32 v30, 63, v30

    // Buffer descriptors (size=4096 bytes, flags=0x20000)
    s_mov_b32 s6, 4096
    s_mov_b32 s7, 0x20000
    s_mov_b32 s10, 4096
    s_mov_b32 s11, 0x20000

    // in_ptr offset = lane * 16 dwords * 4 bytes
    v_lshlrev_b32_e32 v2, 6, v30         // lane * 64 bytes

    // Load A regs (v0..v7) and B regs (v8..v15)
    buffer_load_dwordx4 v[0:3], v2, s[8:11], 0 offen
    v_add_u32_e32 v3, 16, v2
    buffer_load_dwordx4 v[4:7], v3, s[8:11], 0 offen
    v_add_u32_e32 v4, 32, v2
    buffer_load_dwordx4 v[8:11], v4, s[8:11], 0 offen
    v_add_u32_e32 v5, 48, v2
    buffer_load_dwordx4 v[12:15], v5, s[8:11], 0 offen
    s_waitcnt vmcnt(0)

    // Zero accumulators v64..v79
    .irp i, 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
        v_mov_b32_e32 v\i, 0
    .endr

    // MFMA: A=v[0:7], B=v[8:15]
    v_mfma_f32_32x32x64_f8f6f4 v[64:79], v[0:7], v[8:15], v[64:79]

    // Store 16 floats per lane
    v_lshlrev_b32_e32 v6, 6, v30         // lane * 64 bytes
    buffer_store_dwordx4 v[64:67], v6, s[4:7], 0 offen
    v_add_u32_e32 v7, 16, v6
    buffer_store_dwordx4 v[68:71], v7, s[4:7], 0 offen
    v_add_u32_e32 v7, 32, v6
    buffer_store_dwordx4 v[72:75], v7, s[4:7], 0 offen
    v_add_u32_e32 v7, 48, v6
    buffer_store_dwordx4 v[76:79], v7, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _fwd_fp8_mfma_map_debug
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 12
    .amdhsa_accum_offset 220
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _fwd_fp8_mfma_map_debug
    .symbol: _fwd_fp8_mfma_map_debug.kd
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 48
    .max_flat_workgroup_size: 64
    .args:
      - {.name: out_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: in_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata

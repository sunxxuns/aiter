#!/usr/bin/env python3
"""Test QK MFMA output directly"""

import torch
import ctypes
import subprocess

CWD = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"

# Build kernel that outputs S instead of O
asm_code = """
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl _ZN5aiter10test_qk_onlyE
.p2align 8
.type _ZN5aiter10test_qk_onlyE,@function

_ZN5aiter10test_qk_onlyE:
    s_mov_b64 exec, -1
    
    // Thread decomposition (256 threads)
    v_lshrrev_b32_e32 v1, 6, v0           // wave_id
    v_and_b32_e32 v0, 63, v0              // lane_id
    
    // Load args
    s_load_dwordx2 s[4:5], s[0:1], 0x00    // O (S output for debug)
    s_load_dwordx2 s[8:9], s[0:1], 0x08    // Q
    s_load_dwordx2 s[12:13], s[0:1], 0x10  // K
    
    s_waitcnt lgkmcnt(0)
    
    // Buffer descriptors
    s_mov_b32 s10, -1
    s_mov_b32 s11, 0x20000
    s_mov_b32 s14, -1
    s_mov_b32 s15, 0x20000
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // Load Q to LDS (offset 0)
    v_lshlrev_b32_e32 v2, 6, v1
    v_add_u32_e32 v2, v0, v2
    v_lshlrev_b32_e32 v2, 4, v2
    s_mov_b32 m0, 0
    buffer_load_dwordx4 v2, s[8:11], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // Load K to LDS (offset 4096)
    v_lshlrev_b32_e32 v2, 6, v1
    v_add_u32_e32 v2, v0, v2
    v_lshlrev_b32_e32 v2, 4, v2
    s_mov_b32 m0, 4096
    buffer_load_dwordx4 v2, s[12:15], 0 offen lds
    
    s_waitcnt vmcnt(0)
    s_barrier
    
    // QK MFMA (only wave 0)
    v_cmp_eq_u32_e32 vcc, 0, v1
    s_and_saveexec_b64 s[20:21], vcc
    s_cbranch_execz SKIP
    
    // Init S accumulator
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
    
    // LDS addresses
    v_and_b32_e32 v2, 31, v0
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshlrev_b32_e32 v5, 7, v2           // row * 128
    v_lshlrev_b32_e32 v4, 3, v3           // half * 8
    v_add_u32_e32 v5, v5, v4              // Q base
    v_add_u32_e32 v6, 4096, v5            // K base
    
    // One MFMA pass (k=0-15)
    ds_read_b64 v[20:21], v6              // Read K
    ds_read_b64 v[22:23], v5              // Read Q
    s_waitcnt lgkmcnt(0)
    v_accvgpr_write_b32 a0, v20
    v_accvgpr_write_b32 a1, v21
    s_nop 1
    v_mfma_f32_32x32x16_fp8_fp8 v[32:47], a[0:1], v[22:23], v[32:47]
    s_nop 15
    
    // Store S (v32-v47) to output
    v_and_b32_e32 v50, 31, v0
    v_lshrrev_b32_e32 v51, 5, v0
    v_lshlrev_b32_e32 v52, 4, v51         // row_base = half * 16
    
    // Store 16 values per lane
    .macro STORE_S vreg, row_idx
        v_add_u32_e32 v53, \\row_idx, v52
        v_lshlrev_b32_e32 v54, 7, v53     // row * 32 * 4 = row * 128
        v_lshlrev_b32_e32 v55, 2, v50     // col * 4
        v_add_u32_e32 v54, v54, v55
        buffer_store_dword \\vreg, v54, s[4:7], 0 offen
    .endm
    
    STORE_S v32, 0
    STORE_S v33, 1
    STORE_S v34, 2
    STORE_S v35, 3
    STORE_S v36, 4
    STORE_S v37, 5
    STORE_S v38, 6
    STORE_S v39, 7
    STORE_S v40, 8
    STORE_S v41, 9
    STORE_S v42, 10
    STORE_S v43, 11
    STORE_S v44, 12
    STORE_S v45, 13
    STORE_S v46, 14
    STORE_S v47, 15

SKIP:
    s_mov_b64 exec, -1
    s_waitcnt vmcnt(0)
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel _ZN5aiter10test_qk_onlyE
    .amdhsa_group_segment_fixed_size 12288
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 24
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter10test_qk_onlyE
    .symbol: _ZN5aiter10test_qk_onlyE.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 64
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - {.name: ptr_S, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_Q, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global}
      - {.name: ptr_K, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata
"""

# Write and build
with open(f"{CWD}/test_qk_only.s", "w") as f:
    f.write(asm_code)

subprocess.run(["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                "-mcpu=gfx950", "-c", "test_qk_only.s", "-o", "test_qk_only.o"],
               cwd=CWD, check=True)
subprocess.run(["ld.lld", "-shared", "-o", "test_qk_only.co", "test_qk_only.o"],
               cwd=CWD, check=True)

# Load and run
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
module = ctypes.c_void_p()
hip.hipModuleLoad(ctypes.byref(module), f"{CWD}/test_qk_only.co".encode())
func = ctypes.c_void_p()
hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter10test_qk_onlyE")

# Create inputs
seq_len = 32
HD = 128
torch.manual_seed(42)
Q = torch.randn(seq_len, HD, device='cuda').to(torch.float8_e4m3fn)
K = torch.randn(seq_len, HD, device='cuda').to(torch.float8_e4m3fn)
S = torch.zeros(seq_len, seq_len, dtype=torch.float32, device='cuda')  # 32x32

args = [ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr())]
args_arr = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 12288, None, args_arr, None)
hip.hipDeviceSynchronize()

print("QK MFMA test:")
print(f"S has NaN: {torch.isnan(S).any().item()}")
print(f"S[0,:8]: {S[0,:8].tolist()}")

# Reference (one pass, k=0-15)
Q_f32 = Q.float()
K_f32 = K.float()
S_ref_full = torch.matmul(Q_f32[:, :16], K_f32[:, :16].T)  # First 16 cols only
print(f"S_ref[0,:8]: {S_ref_full[0,:8].tolist()}")

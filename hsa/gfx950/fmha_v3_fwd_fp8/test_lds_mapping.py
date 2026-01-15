#!/usr/bin/env python3
"""
GPU test to verify LDS write/read mapping consistency.

Creates a minimal kernel that:
1. Writes data to LDS using our tid/8 mapping
2. Reads data back using MFMA-style mapping
3. Shows the mismatch
"""

import torch
import ctypes
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

# Assembly kernel that demonstrates the mapping mismatch
ASM_CODE = '''
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.set LDS_SIZE, 4352

.globl _ZN5aiter16test_lds_mappingE
.p2align 8
.type _ZN5aiter16test_lds_mappingE,@function

_ZN5aiter16test_lds_mappingE:
    s_mov_b64 exec, -1
    
    // Load output pointer
    s_load_dwordx2 s[4:5], s[0:1], 0x0
    s_waitcnt lgkmcnt(0)
    s_mov_b32 s6, -1
    s_mov_b32 s7, 0x20000
    
    // tid
    v_and_b32_e32 v60, 63, v0       // lane = tid % 64
    
    // ============================================
    // WRITE: Use tid/8 mapping (our current approach)
    // ============================================
    v_lshrrev_b32_e32 v1, 3, v0     // write_row = tid / 8
    v_and_b32_e32 v2, 7, v0         // write_col_idx = tid % 8
    v_lshlrev_b32_e32 v2, 4, v2     // write_col = col_idx * 16
    
    // Write address with pitch-136
    v_mov_b32_e32 v3, 136
    v_mul_lo_u32 v4, v1, v3         // row * 136
    v_add_u32_e32 v4, v4, v2        // + col
    
    // Write pattern: each position gets its (row * 256 + col)
    // This lets us verify which position we actually read
    v_lshlrev_b32_e32 v10, 8, v1    // row * 256
    v_add_u32_e32 v10, v10, v2      // + col
    v_mov_b32_e32 v11, v10
    v_mov_b32_e32 v12, v10
    v_mov_b32_e32 v13, v10
    
    // Write 16 bytes (4 dwords) to LDS
    ds_write_b128 v4, v[10:13]
    
    s_waitcnt lgkmcnt(0)
    s_barrier
    
    // ============================================
    // READ: Use MFMA-style mapping
    // ============================================
    // MFMA row = lane % 32 (simplified for this test)
    v_and_b32_e32 v20, 31, v60      // mfma_row = lane % 32
    
    // MFMA k_offset: lanes 0-31 get k=0, lanes 32-63 get k=8
    v_cmp_ge_u32_e64 vcc, v60, 32
    v_cndmask_b32_e64 v21, 0, 8, vcc  // k_off = 8 if lane >= 32
    
    // Read address with pitch-136
    v_mul_lo_u32 v22, v20, v3       // mfma_row * 136
    v_add_u32_e32 v22, v22, v21     // + k_off
    
    // Read 8 bytes from LDS
    ds_read_b64 v[30:31], v22
    s_waitcnt lgkmcnt(0)
    
    // ============================================
    // OUTPUT: Write results to global memory
    // ============================================
    // Output layout: [tid] = {write_row, write_col, mfma_row, k_off, read_val}
    v_lshlrev_b32_e32 v40, 5, v0    // offset = tid * 32 bytes
    
    // Store write info
    v_mov_b32_e32 v50, v1           // write_row
    v_mov_b32_e32 v51, v2           // write_col
    v_mov_b32_e32 v52, v20          // mfma_row  
    v_mov_b32_e32 v53, v21          // k_off
    buffer_store_dwordx4 v[50:53], v40, s[4:7], 0 offen
    
    // Store read value (2 dwords)
    v_add_u32_e32 v41, 16, v40
    buffer_store_dwordx2 v[30:31], v41, s[4:7], 0 offen
    
    s_endpgm

.p2align 6
.amdhsa_kernel _ZN5aiter16test_lds_mappingE
    .amdhsa_group_segment_fixed_size 4352
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: _ZN5aiter16test_lds_mappingE
    .symbol: _ZN5aiter16test_lds_mappingE.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 4352
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 16
    .vgpr_count: 64
    .max_flat_workgroup_size: 64
    .args:
      - {.name: out_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global}
...
.end_amdgpu_metadata
'''

def main():
    print("=" * 70)
    print("GPU TEST: LDS Write/Read Mapping Mismatch")
    print("=" * 70)
    
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    _ = torch.zeros(1, device='cuda')
    
    # Compile kernel
    asm_file = "/tmp/test_lds_mapping.s"
    co_file = "/tmp/test_lds_mapping.co"
    
    with open(asm_file, 'w') as f:
        f.write(ASM_CODE)
    
    import subprocess
    result = subprocess.run([
        '/opt/rocm/llvm/bin/clang', '-x', 'assembler',
        '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx950',
        '-o', co_file, asm_file
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        return
    
    # Load and run kernel
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter16test_lds_mappingE")
    
    # Output buffer: 64 threads * 8 dwords each
    out = torch.zeros(64 * 8, dtype=torch.int32, device='cuda')
    
    args = [ctypes.c_void_p(out.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 1)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 4352, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # Parse results
    out = out.cpu().numpy().reshape(64, 8)
    
    print("\nResults for first 16 threads:")
    print(f"{'TID':>4} | {'Write Row':>9} | {'Write Col':>9} | {'MFMA Row':>8} | {'k_off':>5} | {'Read Val':>10} | {'Expected':>10} | {'Match':>6}")
    print("-" * 85)
    
    matches = 0
    for tid in range(16):
        write_row = out[tid, 0]
        write_col = out[tid, 1]
        mfma_row = out[tid, 2]
        k_off = out[tid, 3]
        read_val = out[tid, 4]
        
        # Expected: what we wrote at (mfma_row, k_off) position
        expected = mfma_row * 256 + k_off
        match = "YES" if read_val == expected else "NO"
        if read_val == expected:
            matches += 1
        
        print(f"{tid:>4} | {write_row:>9} | {write_col:>9} | {mfma_row:>8} | {k_off:>5} | {read_val:>10} | {expected:>10} | {match:>6}")
    
    print(f"\nMatches: {matches}/16")
    
    # Show lanes 32-47 (different k_off)
    print("\nResults for threads 32-47 (k_off=8):")
    print(f"{'TID':>4} | {'Write Row':>9} | {'MFMA Row':>8} | {'k_off':>5} | {'Read Val':>10} | {'Expected':>10} | {'Match':>6}")
    print("-" * 70)
    
    for tid in range(32, 48):
        write_row = out[tid, 0]
        mfma_row = out[tid, 2]
        k_off = out[tid, 3]
        read_val = out[tid, 4]
        expected = mfma_row * 256 + k_off
        match = "YES" if read_val == expected else "NO"
        print(f"{tid:>4} | {write_row:>9} | {mfma_row:>8} | {k_off:>5} | {read_val:>10} | {expected:>10} | {match:>6}")
    
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
The mismatch shows:
- Thread 0 writes to row 0, but reads from MFMA row 0 (matches)
- Thread 1 writes to row 0, but reads from MFMA row 1 (MISMATCH!)
- Thread 8 writes to row 1, but reads from MFMA row 8 (MISMATCH!)

This is why diagonal padding fails:
- We add (write_row % 2) * 4 to write address
- We add (mfma_row % 2) * 4 to read address
- But write_row != mfma_row, so offsets don't match!

To fix this, either:
1. Use MFMA-compatible mapping for writes (complex)
2. Use XOR swizzle that's row-index agnostic
3. Accept 2-way conflicts with pitch-136
""")

if __name__ == "__main__":
    main()

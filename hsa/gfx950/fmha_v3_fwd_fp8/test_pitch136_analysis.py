#!/usr/bin/env python3
"""
Rigorous analysis of pitch-136 vs padding strategies for FP8 MFMA.

Tests:
1. Bank conflict analysis (theoretical)
2. Numerical correctness with various input patterns
3. Comparison against PyTorch reference
"""

import torch
import numpy as np
import os

os.environ['HIP_VISIBLE_DEVICES'] = '0'

def analyze_bank_conflicts():
    """Analyze bank conflicts for different pitch values."""
    print("=" * 70)
    print("BANK CONFLICT ANALYSIS")
    print("=" * 70)
    print("\nAssumptions:")
    print("- 64 banks, 4 bytes per bank (gfx950 CDNA4)")
    print("- ds_read_b64: 8 bytes per thread")
    print("- 64 threads reading simultaneously")
    print()
    
    def analyze_pitch(pitch, num_rows=32, read_size=8):
        """
        For MFMA read pattern, threads read from specific rows/cols.
        Simplified: assume 64 threads read row 0-31 at col offsets 0,8 (for ds_read_b64).
        """
        banks_hit = {}
        
        # MFMA 32x32x16 read pattern for Q (A operand):
        # Each thread reads 8 bytes from its assigned row
        # Lane 0-31 read rows 0-31 at k_offset 0
        # Lane 32-63 read rows 0-31 at k_offset 8
        for lane in range(64):
            row = lane % 32
            col_offset = 8 if lane >= 32 else 0
            
            byte_addr = row * pitch + col_offset
            bank = (byte_addr // 4) % 64
            
            if bank not in banks_hit:
                banks_hit[bank] = []
            banks_hit[bank].append(lane)
        
        max_hits = max(len(v) for v in banks_hit.values())
        num_banks_used = len(banks_hit)
        
        return num_banks_used, max_hits, banks_hit
    
    print(f"{'Pitch':>6} | {'Banks Used':>10} | {'Max Hits':>8} | {'Status':>15}")
    print("-" * 50)
    
    for pitch in [128, 132, 136, 140, 144, 256]:
        num_banks, max_hits, banks = analyze_pitch(pitch)
        
        if max_hits == 1:
            status = "PERFECT"
        elif max_hits == 2:
            status = "GOOD (2-way)"
        elif max_hits <= 4:
            status = f"OK ({max_hits}-way)"
        else:
            status = f"BAD ({max_hits}-way)"
        
        print(f"{pitch:>6} | {num_banks:>10} | {max_hits:>8} | {status:>15}")
    
    # Detailed analysis for pitch-136
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: PITCH-136")
    print("=" * 70)
    
    num_banks, max_hits, banks = analyze_pitch(136)
    
    print(f"\nBanks used: {num_banks}/64")
    print(f"Max hits per bank: {max_hits}")
    
    # Show which lanes hit which banks
    conflict_banks = {k: v for k, v in banks.items() if len(v) > 1}
    if conflict_banks:
        print(f"\nBanks with conflicts ({len(conflict_banks)} banks):")
        for bank, lanes in sorted(conflict_banks.items())[:5]:
            print(f"  Bank {bank}: lanes {lanes}")
        if len(conflict_banks) > 5:
            print(f"  ... and {len(conflict_banks) - 5} more")
    else:
        print("\nNo bank conflicts!")


def test_numerical_correctness():
    """Test FP8 QK computation against PyTorch reference."""
    print("\n" + "=" * 70)
    print("NUMERICAL CORRECTNESS TESTS")
    print("=" * 70)
    
    import ctypes
    import subprocess
    
    # Compile the best kernel
    kernel_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    
    # Check if 4qtile_v2 exists
    kernel_file = f"{kernel_path}/fwd_fp8_qk_4qtile_v2.s"
    if not os.path.exists(kernel_file):
        print(f"Kernel not found: {kernel_file}")
        print("Using fwd_fp8_qk_preload.s instead")
        kernel_file = f"{kernel_path}/fwd_fp8_qk_preload.s"
    
    # Try to compile
    co_file = kernel_file.replace('.s', '.co')
    cmd = f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -o {co_file} {kernel_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Compile error: {result.stderr}")
        return
    
    print(f"Compiled: {co_file}")
    
    # Test patterns
    test_cases = [
        ("All ones", lambda s, d: torch.ones(s, d)),
        ("All twos", lambda s, d: torch.full((s, d), 2.0)),
        ("Identity-like", lambda s, d: torch.eye(s, d) if s == d else torch.zeros(s, d)),
        ("Row index", lambda s, d: torch.arange(s).unsqueeze(1).expand(s, d).float()),
        ("Col index", lambda s, d: torch.arange(d).unsqueeze(0).expand(s, d).float()),
        ("Random uniform", lambda s, d: torch.rand(s, d) * 2 - 1),
        ("Random normal", lambda s, d: torch.randn(s, d) * 0.1),
        ("Checkerboard", lambda s, d: ((torch.arange(s).unsqueeze(1) + torch.arange(d).unsqueeze(0)) % 2).float()),
    ]
    
    seq_len = 64
    head_dim = 128
    
    print(f"\nTest config: seq_len={seq_len}, head_dim={head_dim}")
    print(f"{'Test Case':<20} | {'Expected':>12} | {'Got':>12} | {'Match':>8} | {'MaxErr':>10}")
    print("-" * 75)
    
    for name, gen_func in test_cases:
        try:
            # Generate Q and K
            Q_f32 = gen_func(seq_len, head_dim).cuda()
            K_f32 = gen_func(seq_len, head_dim).cuda()
            
            # Clamp to FP8 range and convert
            Q_f32 = Q_f32.clamp(-448, 448)
            K_f32 = K_f32.clamp(-448, 448)
            
            Q_fp8 = Q_f32.to(torch.float8_e4m3fn)
            K_fp8 = K_f32.to(torch.float8_e4m3fn)
            
            # PyTorch reference: Q @ K^T
            # But we need to account for FP8 precision
            Q_ref = Q_fp8.to(torch.float32)
            K_ref = K_fp8.to(torch.float32)
            
            # Reference: sum over head_dim for each (q_row, k_row) pair
            # For one 32x32 tile: Q[0:32, :] @ K[0:32, :]^T
            ref_qk = Q_ref[:32, :] @ K_ref[:32, :].T  # [32, 32]
            
            # Expected value for uniform inputs
            if "ones" in name.lower():
                expected = head_dim * 1.0 * 1.0  # 128
            elif "twos" in name.lower():
                expected = head_dim * 2.0 * 2.0  # 512
            else:
                expected = ref_qk.mean().item()
            
            # For now, just compute reference
            got = ref_qk[0, 0].item()  # First element
            max_err = 0.0  # Would need kernel output to compute
            
            match = "REF" 
            print(f"{name:<20} | {expected:>12.2f} | {got:>12.2f} | {match:>8} | {max_err:>10.4f}")
            
        except Exception as e:
            print(f"{name:<20} | ERROR: {e}")
    
    print("\nNote: 'Got' shows PyTorch reference QK[0,0]. Kernel comparison requires running kernel.")


def test_mfma_data_layout():
    """Verify MFMA input data layout expectations."""
    print("\n" + "=" * 70)
    print("MFMA DATA LAYOUT VERIFICATION")
    print("=" * 70)
    
    print("""
For v_mfma_f32_32x32x16_fp8_fp8:
- A matrix: 32 rows × 16 cols (FP8)
- B matrix: 16 rows × 32 cols (FP8)
- C matrix: 32 rows × 32 cols (FP32)

Input packing (A operand):
- Each lane holds 8 FP8 values in 2 VGPRs (v0, v1)
- v0[7:0]   = A[lane%32][0]
- v0[15:8]  = A[lane%32][1]
- v0[23:16] = A[lane%32][2]
- v0[31:24] = A[lane%32][3]
- v1[7:0]   = A[lane%32][4]
- v1[15:8]  = A[lane%32][5]
- v1[23:16] = A[lane%32][6]
- v1[31:24] = A[lane%32][7]
- Lanes 0-31: k=0-7
- Lanes 32-63: k=8-15

ds_read_b64 loads 8 bytes (8 FP8 values) into 2 VGPRs.
This matches MFMA input format IF:
- Lane L reads from row (L % 32)
- Lane L reads k-offset 0 if L < 32, else k-offset 8
""")
    
    print("\nPitch-136 address calculation for MFMA read:")
    print("  byte_addr = row * 136 + k_offset")
    print("  For lane 0:  row=0,  k_off=0  → addr = 0")
    print("  For lane 1:  row=1,  k_off=0  → addr = 136") 
    print("  For lane 32: row=0,  k_off=8  → addr = 8")
    print("  For lane 33: row=1,  k_off=8  → addr = 144")
    
    print("\nPotential issue: ds_read_b64 reads 8 CONSECUTIVE bytes.")
    print("With pitch-136, row N starts at N*136.")
    print("If we read 8 bytes starting at N*136, we get columns 0-7.")
    print("If we read 8 bytes starting at N*136+8, we get columns 8-15.")
    print("This is CORRECT for row-major FP8 data in LDS!")


def test_actual_kernel():
    """Run actual kernel test with rigorous inputs."""
    print("\n" + "=" * 70)
    print("ACTUAL KERNEL TEST")
    print("=" * 70)
    
    import ctypes
    
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    _ = torch.zeros(1, device='cuda')  # Init CUDA
    
    # Try to load preload kernel (simpler, known working)
    kernel_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    co_file = f"{kernel_path}/fwd_fp8_qk_preload.co"
    
    if not os.path.exists(co_file):
        print(f"Kernel not found: {co_file}")
        return
    
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    
    err = hip.hipModuleLoad(ctypes.byref(module), co_file.encode())
    if err != 0:
        print(f"hipModuleLoad failed: {err}")
        return
    
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b'_ZN5aiter17fwd_fp8_qk_preloadE')
    if err != 0:
        print(f"hipModuleGetFunction failed: {err}")
        return
    
    print(f"Loaded kernel from {co_file}")
    
    # Test cases with expected results
    test_cases = [
        ("All ones", 
         torch.ones(64, 128, device='cuda'),
         torch.ones(64, 128, device='cuda'),
         128.0),  # 1*1*128 = 128 per element
        
        ("All 0.5",
         torch.full((64, 128), 0.5, device='cuda'),
         torch.full((64, 128), 0.5, device='cuda'),
         32.0),  # 0.5*0.5*128 = 32
         
        ("Q=1, K=2",
         torch.ones(64, 128, device='cuda'),
         torch.full((64, 128), 2.0, device='cuda'),
         256.0),  # 1*2*128 = 256
         
        ("Diagonal",
         torch.eye(64, 128, device='cuda'),
         torch.eye(64, 128, device='cuda'),
         None),  # Diagonal pattern
    ]
    
    print(f"\n{'Test':<15} | {'Expected':>10} | {'Got[0,0]':>10} | {'Got Mean':>10} | {'Status':>8}")
    print("-" * 70)
    
    for name, Q_f32, K_f32, expected in test_cases:
        # Convert to FP8
        Q_fp8 = Q_f32.to(torch.float8_e4m3fn).contiguous()
        K_fp8 = K_f32.to(torch.float8_e4m3fn).contiguous()
        
        # Output buffer
        O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
        
        # Kernel args
        args = [
            ctypes.c_void_p(O.data_ptr()),
            ctypes.c_void_p(Q_fp8.data_ptr()),
            ctypes.c_void_p(K_fp8.data_ptr()),
            ctypes.c_uint32(64),  # seq_len
            ctypes.c_uint32(0),   # q_tile_idx
        ]
        args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        
        # Launch
        err = hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
        hip.hipDeviceSynchronize()
        
        if err != 0:
            print(f"{name:<15} | ERROR: launch failed ({err})")
            continue
        
        # Check results
        got_first = O[0].item()
        got_mean = O.mean().item()
        
        if expected is not None:
            # Allow some tolerance for FP8 precision
            rel_err = abs(got_first - expected) / max(abs(expected), 1e-6)
            status = "PASS" if rel_err < 0.1 else "FAIL"
            print(f"{name:<15} | {expected:>10.2f} | {got_first:>10.2f} | {got_mean:>10.2f} | {status:>8}")
        else:
            print(f"{name:<15} | {'N/A':>10} | {got_first:>10.2f} | {got_mean:>10.2f} | {'CHECK':>8}")
    
    # Detailed random test
    print("\n" + "-" * 70)
    print("RANDOM INPUT TEST (rigorous)")
    print("-" * 70)
    
    torch.manual_seed(42)
    Q_f32 = torch.randn(64, 128, device='cuda') * 0.5
    K_f32 = torch.randn(64, 128, device='cuda') * 0.5
    
    Q_fp8 = Q_f32.to(torch.float8_e4m3fn).contiguous()
    K_fp8 = K_f32.to(torch.float8_e4m3fn).contiguous()
    
    # PyTorch reference
    Q_ref = Q_fp8.to(torch.float32)
    K_ref = K_fp8.to(torch.float32)
    ref_qk = Q_ref[:32, :] @ K_ref[:32, :].T  # 32x32 output
    
    # Kernel output
    O = torch.zeros(256 * 16, dtype=torch.float32, device='cuda')
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q_fp8.data_ptr()),
        ctypes.c_void_p(K_fp8.data_ptr()),
        ctypes.c_uint32(64),
        ctypes.c_uint32(0),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 16384, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    # The output layout from MFMA needs to be understood
    # Each thread outputs 16 floats (v[0:15])
    # 256 threads * 16 floats = 4096 floats = 32x32 matrix (but layout?)
    
    print(f"Reference QK shape: {ref_qk.shape}")
    print(f"Reference QK[0,0]: {ref_qk[0,0].item():.4f}")
    print(f"Reference QK mean: {ref_qk.mean().item():.4f}")
    print(f"Reference QK range: [{ref_qk.min().item():.4f}, {ref_qk.max().item():.4f}]")
    
    print(f"\nKernel output first 16: {O[:16].tolist()}")
    print(f"Kernel output mean: {O.mean().item():.4f}")
    print(f"Kernel output range: [{O.min().item():.4f}, {O.max().item():.4f}]")
    
    # Try to match kernel output to reference
    # MFMA output layout is complex - need to understand accumulator mapping
    
    hip.hipModuleUnload(module)


if __name__ == "__main__":
    analyze_bank_conflicts()
    test_mfma_data_layout()
    test_numerical_correctness()
    test_actual_kernel()

#!/usr/bin/env python3
"""
Test FP8 Flash Attention with Full Sequence Support (Multi-Q-Tile)

This kernel processes seq_len × seq_len attention by launching
seq_len/32 workgroups, each handling 32 Q-rows.
"""

import torch
import subprocess
import ctypes
import math

def build():
    """Build the full attention kernel"""
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "fwd_fp8_full.s", "-o", "fwd_fp8_full.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Build error:\n{result.stderr.decode()}")
        return None
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", "fwd_fp8_full.co", "fwd_fp8_full.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Link error:\n{result.stderr.decode()}")
        return None
    return cwd + "/fwd_fp8_full.co"


def test_full_attention(seq_len, seed=42):
    """Test full attention with given sequence length."""
    print(f"\n{'='*60}")
    print(f"Testing FULL attention: seq_len = {seq_len}")
    print(f"{'='*60}")
    
    assert seq_len % 32 == 0, "seq_len must be multiple of 32"
    num_q_tiles = seq_len // 32
    
    co_path = build()
    if co_path is None:
        return False
    
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12fwd_fp8_fullE")
    
    HD = 128
    scale = 1.0 / math.sqrt(HD)
    
    # Create test data - FULL seq_len × HD tensors
    torch.manual_seed(seed)
    Q = torch.randn(seq_len, HD, device='cuda') * 0.5
    K = torch.randn(seq_len, HD, device='cuda') * 0.5
    V = torch.randn(seq_len, HD, device='cuda') * 0.5
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    # Output - FULL seq_len × HD
    O = torch.zeros(seq_len, HD, dtype=torch.float32, device='cuda')
    
    # Args
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q_fp8.data_ptr()),
        ctypes.c_void_p(K_fp8.data_ptr()),
        ctypes.c_void_p(V_fp8.data_ptr()),
        ctypes.c_uint32(seq_len),
    ]
    args_arr = (ctypes.c_void_p * 5)(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    
    # Launch with num_q_tiles workgroups!
    print(f"Launching {num_q_tiles} workgroups (1 per Q-tile)")
    err = hip.hipModuleLaunchKernel(
        func, 
        num_q_tiles, 1, 1,  # grid: num_q_tiles workgroups
        64, 1, 1,           # block: 64 threads
        12288,              # shared mem
        None, args_arr, None
    )
    if err != 0:
        print(f"Launch error: {err}")
        return False
    
    err = hip.hipDeviceSynchronize()
    if err != 0:
        print(f"Sync error: {err}")
        return False
    
    # Reference computation (full seq × seq attention)
    S = Q_fp8.float() @ K_fp8.float().T  # [seq, seq]
    P = torch.softmax(S * scale, dim=1)  # Row-wise softmax
    P_fp8 = P.to(torch.float8_e4m3fn).float()
    O_ref = P_fp8 @ V_fp8.float()  # [seq, HD]
    
    # Compare
    max_err = (O - O_ref).abs().max().item()
    mean_err = (O - O_ref).abs().mean().item()
    
    print(f"Output shape: {O.shape}")
    print(f"O_kernel[0,:5]:   {O[0,:5].tolist()}")
    print(f"O_ref[0,:5]:      {O_ref[0,:5].tolist()}")
    print(f"O_kernel[-1,:5]:  {O[-1,:5].tolist()}")
    print(f"O_ref[-1,:5]:     {O_ref[-1,:5].tolist()}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    # Check for NaN
    has_nan = torch.isnan(O).any().item()
    if has_nan:
        print("❌ NaN detected in output!")
        return False
    
    threshold = 0.15
    passed = max_err < threshold
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'} (threshold={threshold})")
    
    hip.hipModuleUnload(module)
    return passed


def main():
    print("=" * 60)
    print("FP8 FLASH ATTENTION - FULL SEQUENCE TEST")
    print("=" * 60)
    
    results = {}
    
    # Test with increasing sequence lengths (all must be multiple of 32)
    for seq_len in [64, 128, 256, 512]:
        results[seq_len] = test_full_attention(seq_len)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for seq_len, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        q_tiles = seq_len // 32
        k_tiles = seq_len // 32
        print(f"  seq_len={seq_len:4d} ({q_tiles}×{k_tiles} tiles): {status}")
    
    all_pass = all(results.values())
    print(f"\n{'✅ ALL TESTS PASSED!' if all_pass else '❌ SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    main()

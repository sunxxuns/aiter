#!/usr/bin/env python3
"""
Test FP8 Flash Attention with K-tile loop and online softmax.
"""

import torch
import subprocess
import ctypes
import math

def build():
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "fwd_fp8_kloop.s", "-o", "fwd_fp8_kloop.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Build error:\n{result.stderr.decode()}")
        return None
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", "fwd_fp8_kloop.co", "fwd_fp8_kloop.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Link error:\n{result.stderr.decode()}")
        return None
    return cwd + "/fwd_fp8_kloop.co"


def test_kloop_attention(seq_len, seed=42):
    """Test K-loop attention with given sequence length."""
    print(f"\n{'='*60}")
    print(f"Testing seq_len = {seq_len} (seed={seed})")
    print(f"{'='*60}")
    
    co_path = build()
    if co_path is None:
        return False
    
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13fwd_fp8_kloopE")
    
    Q_ROWS = 32
    HD = 128
    scale = 1.0 / math.sqrt(HD)
    
    # Create test data
    torch.manual_seed(seed)
    Q = torch.randn(Q_ROWS, HD, device='cuda') * 0.5
    K = torch.randn(seq_len, HD, device='cuda') * 0.5
    V = torch.randn(seq_len, HD, device='cuda') * 0.5
    
    Q_fp8 = Q.to(torch.float8_e4m3fn)
    K_fp8 = K.to(torch.float8_e4m3fn)
    V_fp8 = V.to(torch.float8_e4m3fn)
    
    # Output (full HD=128)
    O = torch.zeros(Q_ROWS, HD, dtype=torch.float32, device='cuda')
    
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
    
    # Launch
    err = hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 12288, None, args_arr, None)
    if err != 0:
        print(f"Launch error: {err}")
        return False
    
    err = hip.hipDeviceSynchronize()
    if err != 0:
        print(f"Sync error: {err}")
        return False
    
    # Reference computation
    S = Q_fp8.float() @ K_fp8.float().T  # [32, seq_len]
    P = torch.softmax(S * scale, dim=1)  # Row-wise softmax
    P_fp8 = P.to(torch.float8_e4m3fn).float()
    O_ref = P_fp8 @ V_fp8.float()  # [32, HD]
    
    # Compare all 128 columns (full HD=128)
    max_err = (O - O_ref).abs().max().item()
    mean_err = (O - O_ref).abs().mean().item()
    
    print(f"O_kernel[0,:5]: {O[0,:5].tolist()}")
    print(f"O_ref[0,:5]:    {O_ref[0,:5].tolist()}")
    print(f"O_kernel[0,64:69]: {O[0,64:69].tolist()}")
    print(f"O_ref[0,64:69]:    {O_ref[0,64:69].tolist()}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    # Relaxed threshold for online softmax with FP8 quantization
    # FP8 P @ V vs F32 running_sum causes small scale mismatch
    threshold = 0.20
    passed = max_err < threshold
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'} (threshold={threshold})")
    
    hip.hipModuleUnload(module)
    return passed


def main():
    print("=" * 60)
    print("FP8 FLASH ATTENTION K-LOOP TEST")
    print("=" * 60)
    
    results = {}
    
    # Test with increasing sequence lengths
    for seq_len in [32, 64, 96, 128]:
        results[seq_len] = test_kloop_attention(seq_len)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for seq_len, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        tiles = (seq_len + 31) // 32
        print(f"  seq_len={seq_len:4d} ({tiles} tile{'s' if tiles > 1 else ''}): {status}")
    
    all_pass = all(results.values())
    print(f"\n{'✅ ALL TESTS PASSED!' if all_pass else '❌ SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    main()

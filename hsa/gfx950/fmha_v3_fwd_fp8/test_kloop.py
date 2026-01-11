#!/usr/bin/env python3
"""
Test K-loop: Load from multiple K tiles using scalar offset.
"""

import torch
import subprocess
import ctypes

def build():
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "test_kloop.s", "-o", "test_kloop.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Build error:\n{result.stderr.decode()}")
        return None
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", "test_kloop.co", "test_kloop.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Link error:\n{result.stderr.decode()}")
        return None
    return cwd + "/test_kloop.co"


def test_kloop(num_tiles):
    """Test K-loop with given number of tiles."""
    print(f"\n{'='*60}")
    print(f"Testing {num_tiles} K-tiles (seq_len = {num_tiles * 32})")
    print(f"{'='*60}")
    
    co_path = build()
    if co_path is None:
        return False
    
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter10test_kloopE")
    
    # Create K with distinct values per tile
    # Each tile: 32 rows × 128 cols = 4096 floats = 16384 bytes
    tile_size = 32 * 128  # floats per tile
    seq_len = num_tiles * 32
    
    K = torch.zeros(num_tiles * tile_size, dtype=torch.float32, device='cuda')
    for t in range(num_tiles):
        # Set first 64 elements of each tile to (t+1)
        K[t * tile_size : t * tile_size + 64] = float(t + 1)
    
    O = torch.zeros(64, dtype=torch.float32, device='cuda')
    
    print(f"K shape: {K.shape}")
    print(f"K[0:4] (tile 0): {K[0:4].tolist()}")
    if num_tiles > 1:
        print(f"K[{tile_size}:{tile_size+4}] (tile 1): {K[tile_size:tile_size+4].tolist()}")
    
    # Args
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(seq_len),
    ]
    args_arr = (ctypes.c_void_p * 3)(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    
    print(f"\nLaunching...")
    err = hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 0, None, args_arr, None)
    if err != 0:
        print(f"Launch error: {err}")
        return False
    
    err = hip.hipDeviceSynchronize()
    if err != 0:
        print(f"Sync error: {err}")
        return False
    
    # Expected: O[i] = sum of K[t*tile_size + i] across all tiles
    # = sum(t+1 for t in range(num_tiles)) = num_tiles * (num_tiles + 1) / 2
    expected_sum = num_tiles * (num_tiles + 1) / 2
    
    print(f"\nO[0:8] = {O[0:8].tolist()}")
    print(f"Expected: {expected_sum}")
    
    max_err = (O[:64] - expected_sum).abs().max().item()
    print(f"Max error: {max_err}")
    
    passed = max_err < 0.001
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}")
    
    hip.hipModuleUnload(module)
    return passed


def main():
    print("=" * 60)
    print("K-LOOP TEST: Scalar offset advancement")
    print("=" * 60)
    
    results = {}
    for num_tiles in [1, 2, 4, 8]:
        results[num_tiles] = test_kloop(num_tiles)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for num_tiles, passed in results.items():
        seq_len = num_tiles * 32
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {num_tiles} tiles (seq={seq_len}): {status}")
    
    all_pass = all(results.values())
    print(f"\n{'✅ ALL TESTS PASSED!' if all_pass else '❌ SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    main()

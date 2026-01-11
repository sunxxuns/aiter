#!/usr/bin/env python3
"""
Test buffer_load with scalar offset.
"""

import torch
import subprocess
import ctypes

def build():
    cwd = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "debug_bufload.s", "-o", "debug_bufload.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Build error:\n{result.stderr.decode()}")
        return None
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", "debug_bufload.co", "debug_bufload.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Link error:\n{result.stderr.decode()}")
        return None
    return cwd + "/debug_bufload.co"


def test_buffer_load_offset():
    """Test buffer_load with scalar offset."""
    print("=" * 60)
    print("DEBUG: buffer_load with scalar offset")
    print("=" * 60)
    
    co_path = build()
    if co_path is None:
        return False
    
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter12debug_bufloadE")
    
    # K[0:64] = 1.0, K[64:128] = 2.0
    K = torch.zeros(128, dtype=torch.float32, device='cuda')
    K[:64] = 1.0
    K[64:128] = 2.0
    
    O = torch.zeros(64, dtype=torch.float32, device='cuda')
    
    # offset2 = 256 bytes = 64 floats
    offset2 = 256
    
    print(f"\nK ptr: {K.data_ptr():#x}")
    print(f"O ptr: {O.data_ptr():#x}")
    print(f"K[0:8]:   {K[0:8].tolist()}")
    print(f"K[64:72]: {K[64:72].tolist()}")
    print(f"offset2: {offset2} bytes")
    
    # Args: O, K, offset2
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_uint32(offset2),
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
    
    # Expected: O[i] = K[i] + K[64+i] = 1.0 + 2.0 = 3.0
    expected = 3.0
    
    print(f"\nO[0:8] = {O[0:8].tolist()}")
    print(f"Expected: {expected}")
    
    max_err = (O - expected).abs().max().item()
    print(f"Max error: {max_err}")
    
    passed = max_err < 0.001
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}")
    
    hip.hipModuleUnload(module)
    return passed


if __name__ == "__main__":
    test_buffer_load_offset()

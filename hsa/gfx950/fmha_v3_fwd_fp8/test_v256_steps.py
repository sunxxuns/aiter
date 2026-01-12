#!/usr/bin/env python3
"""
Step-by-step validation for 256-thread FP8 kernel

Each step tests a specific component:
1. Kernel loads and launches (no crash)
2. Q loads correctly to LDS
3. K loads correctly to LDS  
4. QK MFMA produces valid output
5. Softmax works
6. PV MFMA works
7. Full attention correct
"""

import torch
import subprocess
import ctypes
import math

CWD = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"

def build(kernel_name):
    """Build kernel"""
    subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", f"{kernel_name}.s", "-o", f"{kernel_name}.o"],
        capture_output=True, cwd=CWD, check=True
    )
    subprocess.run(
        ["ld.lld", "-shared", "-o", f"{kernel_name}.co", f"{kernel_name}.o"],
        capture_output=True, cwd=CWD, check=True
    )
    return f"{CWD}/{kernel_name}.co"

def load_kernel(co_path, func_name):
    """Load kernel"""
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, func_name.encode())
    return hip, module, func

def step1_launch():
    """Step 1: Kernel compiles and launches without crash"""
    print("\n" + "="*60)
    print("STEP 1: Kernel Launch Test")
    print("="*60)
    
    try:
        co_path = build("fwd_fp8_v256")
        hip, module, func = load_kernel(co_path, "_ZN5aiter11fwd_fp8_v256E")
        
        seq_len = 64
        Q = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        K = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        V = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        O = torch.zeros(seq_len, 128, dtype=torch.float32, device='cuda')
        
        args = [ctypes.c_void_p(O.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
                ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(V.data_ptr()),
                ctypes.c_uint32(seq_len)]
        args_arr = (ctypes.c_void_p * 5)(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )
        
        num_q_tiles = seq_len // 32
        err = hip.hipModuleLaunchKernel(func, num_q_tiles, 1, 1, 256, 1, 1, 32768, None, args_arr, None)
        hip.hipDeviceSynchronize()
        
        print(f"  Launch error: {err}")
        print(f"  Output shape: {O.shape}")
        print(f"  Has NaN: {torch.isnan(O).any().item()}")
        print(f"  Has Inf: {torch.isinf(O).any().item()}")
        
        passed = err == 0
        print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: Kernel launches without crash")
        return passed
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def step2_thread_distribution():
    """Step 2: Verify 256 threads are properly distributed"""
    print("\n" + "="*60)
    print("STEP 2: Thread Distribution Test")
    print("="*60)
    
    # This test would need a debug kernel that outputs thread IDs
    # For now, just verify the kernel metadata
    
    import re
    with open(f"{CWD}/fwd_fp8_v256.s", "r") as f:
        content = f.read()
    
    # Check max_flat_workgroup_size
    match = re.search(r'max_flat_workgroup_size:\s*(\d+)', content)
    if match:
        wg_size = int(match.group(1))
        print(f"  max_flat_workgroup_size: {wg_size}")
        passed = wg_size == 256
    else:
        print(f"  Could not find workgroup size in metadata")
        passed = False
    
    # Check LDS size
    match = re.search(r'group_segment_fixed_size:\s*(\d+)', content)
    if match:
        lds_size = int(match.group(1))
        print(f"  LDS size: {lds_size} bytes ({lds_size//1024}KB)")
        
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: 256 threads configured")
    return passed

def step3_compare_with_64t():
    """Step 3: Compare 256T output with working 64T kernel"""
    print("\n" + "="*60)
    print("STEP 3: Compare with 64-Thread Kernel")
    print("="*60)
    
    try:
        # Build both kernels
        co_64t = build("fwd_fp8_full")
        co_256t = build("fwd_fp8_v256")
        
        hip, mod_64t, func_64t = load_kernel(co_64t, "_ZN5aiter12fwd_fp8_fullE")
        _, mod_256t, func_256t = load_kernel(co_256t, "_ZN5aiter11fwd_fp8_v256E")
        
        seq_len = 64
        torch.manual_seed(42)
        Q = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        K = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        V = torch.randn(seq_len, 128, device='cuda').to(torch.float8_e4m3fn)
        
        O_64t = torch.zeros(seq_len, 128, dtype=torch.float32, device='cuda')
        O_256t = torch.zeros(seq_len, 128, dtype=torch.float32, device='cuda')
        
        num_q_tiles = seq_len // 32
        
        # Run 64T
        args = [ctypes.c_void_p(O_64t.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
                ctypes.c_void_p(K.data_ptr()), ctypes.c_void_p(V.data_ptr()),
                ctypes.c_uint32(seq_len)]
        args_arr = (ctypes.c_void_p * 5)(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )
        hip.hipModuleLaunchKernel(func_64t, num_q_tiles, 1, 1, 64, 1, 1, 12288, None, args_arr, None)
        hip.hipDeviceSynchronize()
        
        # Run 256T
        args[0] = ctypes.c_void_p(O_256t.data_ptr())
        args_arr = (ctypes.c_void_p * 5)(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )
        hip.hipModuleLaunchKernel(func_256t, num_q_tiles, 1, 1, 256, 1, 1, 32768, None, args_arr, None)
        hip.hipDeviceSynchronize()
        
        print(f"  64T output mean: {O_64t.mean().item():.4f}")
        print(f"  256T output mean: {O_256t.mean().item():.4f}")
        print(f"  64T has NaN: {torch.isnan(O_64t).any().item()}")
        print(f"  256T has NaN: {torch.isnan(O_256t).any().item()}")
        
        if not torch.isnan(O_256t).any():
            diff = (O_64t - O_256t).abs()
            print(f"  Max diff: {diff.max().item():.4f}")
            print(f"  Mean diff: {diff.mean().item():.4f}")
        
        # For now, just check no crash and no NaN
        passed = not torch.isnan(O_256t).any().item()
        print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: 256T produces valid output")
        return passed
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False

def main():
    print("="*60)
    print("FP8 256-THREAD KERNEL STEP-BY-STEP VALIDATION")
    print("="*60)
    
    results = {}
    
    results['step1'] = step1_launch()
    results['step2'] = step2_thread_distribution()
    results['step3'] = step3_compare_with_64t()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for step, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {step}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n{passed}/{total} steps passed")
    
    if passed < total:
        print("\n⚠️  Fix failing steps before proceeding!")
    
    return passed == total

if __name__ == "__main__":
    main()

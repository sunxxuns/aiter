#!/usr/bin/env python3
"""
Test FP8 256T Step 2: Shared K Load

Verifies:
1. All 256 threads cooperatively load K to shared LDS region
2. K data at corners is correctly read back
"""

import torch
import subprocess
import ctypes
import os

def build():
    """Compile and link the step2 kernel"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "fwd_fp8_256t_step2.s", "-o", "fwd_fp8_256t_step2.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
        
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", "fwd_fp8_256t_step2.co", "fwd_fp8_256t_step2.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Link error:\n{result.stderr.decode()}")
        return None
        
    return os.path.join(cwd, "fwd_fp8_256t_step2.co")

def run_step2_test():
    """Test Step 2: Shared K load"""
    print("=" * 60)
    print("FP8 256T Step 2: Shared K Load Test")
    print("=" * 60)
    
    # Build
    print("\n1. Building kernel...")
    co_path = build()
    if co_path is None:
        return False
    print("   ✅ Build successful")
    
    # Load
    print("\n2. Loading kernel...")
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    err = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if err != 0:
        print(f"   ❌ hipModuleLoad failed: {err}")
        return False
        
    func = ctypes.c_void_p()
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter16fwd_fp8_256t_s2E")
    if err != 0:
        print(f"   ❌ hipModuleGetFunction failed: {err}")
        return False
    print("   ✅ Kernel loaded")
    
    # Create test data
    print("\n3. Creating test data...")
    
    # Q: 32 rows × 128 cols (not really used in this test)
    Q_data = torch.zeros(32, 128, dtype=torch.uint8, device='cuda')
    
    # K: 32 rows × 128 cols with distinct marker values
    K_data = torch.zeros(32, 128, dtype=torch.uint8, device='cuda')
    for row in range(32):
        K_data[row, :] = row + 100  # Different from Q
    
    # Set corner markers
    K_data[0, 0] = 0xAA      # K[0,0]
    K_data[0, 127] = 0xBB    # K[0,127]
    K_data[31, 0] = 0xCC     # K[31,0]
    K_data[31, 127] = 0xDD   # K[31,127]
    
    # Output: 4 floats
    O = torch.zeros(4, dtype=torch.float32, device='cuda')
    
    print(f"   K shape: {K_data.shape}")
    print(f"   K corners: [{K_data[0,0].item():#x}, {K_data[0,127].item():#x}, {K_data[31,0].item():#x}, {K_data[31,127].item():#x}]")
    
    # Launch kernel
    print("\n4. Launching kernel...")
    
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q_data.data_ptr()),
        ctypes.c_void_p(K_data.data_ptr()),
    ]
    args_arr = (ctypes.c_void_p * 3)(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    
    # Launch: 1 block of 256 threads, 8KB shared memory
    err = hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,
        256, 1, 1,
        8192,           # 8KB for Q + K
        None,
        args_arr,
        None
    )
    
    if err != 0:
        print(f"   ❌ Kernel launch failed: {err}")
        return False
    
    err = hip.hipDeviceSynchronize()
    if err != 0:
        print(f"   ❌ hipDeviceSynchronize failed: {err}")
        return False
    
    print("   ✅ Kernel executed")
    
    # Verify results
    print("\n5. Verifying results...")
    
    O_cpu = O.cpu().numpy()
    
    expected = [0xAA, 0xBB, 0xCC, 0xDD]
    labels = ["K[0,0]", "K[0,127]", "K[31,0]", "K[31,127]"]
    
    all_pass = True
    for i in range(4):
        got = int(O_cpu[i])
        exp = expected[i]
        if got == exp:
            print(f"   ✅ {labels[i]} = {got:#x} (expected {exp:#x})")
        else:
            print(f"   ❌ {labels[i]} = {got:#x} (expected {exp:#x})")
            all_pass = False
    
    # Cleanup
    hip.hipModuleUnload(module)
    
    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ STEP 2 PASSED: Shared K load working correctly")
    else:
        print("❌ STEP 2 FAILED: Issues found")
    print("=" * 60)
    
    return all_pass

if __name__ == "__main__":
    success = run_step2_test()
    exit(0 if success else 1)

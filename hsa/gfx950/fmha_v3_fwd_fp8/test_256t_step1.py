#!/usr/bin/env python3
"""
Test FP8 256T Step 1: Wave-Aware Q Load

Verifies:
1. Wave ID extraction works correctly (0-3)
2. Lane ID extraction works correctly (0-63)
3. Each wave loads its 8 Q rows to correct LDS region
4. Q data is correctly read back from LDS
"""

import torch
import subprocess
import ctypes
import os

def build():
    """Compile and link the step1 kernel"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Compile
    result = subprocess.run(
        ["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx950", "-c", "fwd_fp8_256t_step1.s", "-o", "fwd_fp8_256t_step1.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
        
    # Link
    result = subprocess.run(
        ["ld.lld", "-shared", "-o", "fwd_fp8_256t_step1.co", "fwd_fp8_256t_step1.o"],
        capture_output=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"Link error:\n{result.stderr.decode()}")
        return None
        
    return os.path.join(cwd, "fwd_fp8_256t_step1.co")

def run_step1_test():
    """Test Step 1: Wave-aware Q load"""
    print("=" * 60)
    print("FP8 256T Step 1: Wave-Aware Q Load Test")
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
    err = hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter16fwd_fp8_256t_s1E")
    if err != 0:
        print(f"   ❌ hipModuleGetFunction failed: {err}")
        return False
    print("   ✅ Kernel loaded")
    
    # Create test data
    print("\n3. Creating test data...")
    
    # Q: 32 rows × 128 cols, each row has distinct pattern
    Q_data = torch.zeros(32, 128, dtype=torch.uint8, device='cuda')
    for row in range(32):
        Q_data[row, :] = row  # Each row has value = row_index
    
    # Set specific marker values for each wave's first row
    for wave in range(4):
        row = wave * 8
        Q_data[row, 0] = 0x38 + wave * 16      # 0x38, 0x48, 0x58, 0x68
        Q_data[row, 127] = 0x39 + wave * 16    # 0x39, 0x49, 0x59, 0x69
    
    # Output: 256 threads × 4 floats = 1024 floats
    O = torch.zeros(256 * 4, dtype=torch.float32, device='cuda')
    
    print(f"   Q shape: {Q_data.shape}")
    print(f"   Q markers: wave 0 row 0: [{Q_data[0,0].item():#x}, {Q_data[0,127].item():#x}]")
    print(f"              wave 1 row 0: [{Q_data[8,0].item():#x}, {Q_data[8,127].item():#x}]")
    print(f"              wave 2 row 0: [{Q_data[16,0].item():#x}, {Q_data[16,127].item():#x}]")
    print(f"              wave 3 row 0: [{Q_data[24,0].item():#x}, {Q_data[24,127].item():#x}]")
    
    # Launch kernel
    print("\n4. Launching kernel...")
    
    # Args: (ptr_O, ptr_Q)
    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q_data.data_ptr()),
    ]
    args_arr = (ctypes.c_void_p * 2)(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    
    # Launch: 1 block of 256 threads, 4KB shared memory
    err = hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,        # grid
        256, 1, 1,      # block
        4096,           # shared memory
        None,           # stream
        args_arr,       # args
        None            # extra
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
    
    O_cpu = O.reshape(256, 4).cpu().numpy()
    
    all_pass = True
    
    # Check each wave
    for wave_id in range(4):
        print(f"\n   Wave {wave_id}:")
        
        # Sample thread 0 of each wave
        tid = wave_id * 64
        out_wave_id = O_cpu[tid, 0]
        out_lane_id = O_cpu[tid, 1]
        out_q0 = O_cpu[tid, 2]
        out_q127 = O_cpu[tid, 3]
        
        expected_q0 = float(0x38 + wave_id * 16)
        expected_q127 = float(0x39 + wave_id * 16)
        
        print(f"     Thread {tid}: wave_id={out_wave_id:.0f}, lane_id={out_lane_id:.0f}")
        print(f"     Q[0]={out_q0:.0f} (expected {expected_q0:.0f}), Q[127]={out_q127:.0f} (expected {expected_q127:.0f})")
        
        # Verify wave_id
        if out_wave_id != wave_id:
            print(f"     ❌ Wave ID mismatch")
            all_pass = False
        else:
            print(f"     ✅ Wave ID correct")
        
        # Verify lane_id (should be 0)
        if out_lane_id != 0:
            print(f"     ❌ Lane ID mismatch")
            all_pass = False
        else:
            print(f"     ✅ Lane ID correct")
        
        # Verify Q values
        if out_q0 != expected_q0:
            print(f"     ❌ Q[0] mismatch")
            all_pass = False
        else:
            print(f"     ✅ Q[0] correct (wave loaded to correct LDS region)")
            
        if out_q127 != expected_q127:
            print(f"     ❌ Q[127] mismatch")
            all_pass = False
        else:
            print(f"     ✅ Q[127] correct")
    
    # Check all 256 threads for wave_id and lane_id
    print("\n   Checking all 256 threads...")
    wave_id_errors = 0
    lane_id_errors = 0
    
    for tid in range(256):
        expected_wave = tid // 64
        expected_lane = tid % 64
        
        if O_cpu[tid, 0] != expected_wave:
            wave_id_errors += 1
            if wave_id_errors <= 3:
                print(f"     Wave error tid={tid}: got {O_cpu[tid, 0]}, expected {expected_wave}")
        if O_cpu[tid, 1] != expected_lane:
            lane_id_errors += 1
            if lane_id_errors <= 3:
                print(f"     Lane error tid={tid}: got {O_cpu[tid, 1]}, expected {expected_lane}")
    
    if wave_id_errors == 0:
        print(f"     ✅ All 256 wave_id values correct")
    else:
        print(f"     ❌ {wave_id_errors} wave_id errors")
        all_pass = False
        
    if lane_id_errors == 0:
        print(f"     ✅ All 256 lane_id values correct")
    else:
        print(f"     ❌ {lane_id_errors} lane_id errors")
        all_pass = False
    
    # Cleanup
    hip.hipModuleUnload(module)
    
    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ STEP 1 PASSED: Wave-aware Q load working correctly")
    else:
        print("❌ STEP 1 FAILED: Issues found")
    print("=" * 60)
    
    return all_pass

if __name__ == "__main__":
    success = run_step1_test()
    exit(0 if success else 1)

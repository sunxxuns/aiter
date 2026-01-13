#!/usr/bin/env python3
"""
Understand TR_B8 behavior exactly.
LDS filled with LDS[i] = i, then read with TR_B8 at various base addresses.
"""

import torch
import subprocess
import ctypes
import numpy as np
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'exp_tr_b8.s', '-o', 'exp_tr_b8.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'exp_tr_b8.co', 'exp_tr_b8.o'], cwd=cwd)
    return str(cwd / 'exp_tr_b8.co')

def run_test(hip, func, base_addr):
    out = torch.zeros(64 * 2, dtype=torch.int32, device='cuda')  # 64 lanes × 2 dwords
    
    # Args: output ptr, base_addr
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_uint32(base_addr)]
    args_ptrs = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    return out.cpu().numpy()

def analyze_output(out, base_addr):
    """Analyze what TR_B8 did"""
    print(f"\nBase addr = {base_addr}")
    print("-" * 60)
    
    # Convert to bytes view
    out_bytes = out.view(np.uint8).reshape(64, 8)
    
    # Show key lanes to understand pattern
    print("Lane  | Byte[0] (all 8 bytes are identical)")
    print("-" * 40)
    for lane in range(64):
        b0 = out_bytes[lane][0]
        if lane < 16 or lane >= 48:
            print(f"{lane:4d}  | {b0:3d}")
        elif lane == 16:
            print("  ... (lanes 16-47)")
    
    # Summary: what byte does each lane get?
    byte0_all = [out_bytes[lane][0] for lane in range(64)]
    print(f"\nAll 64 lanes byte[0]: {byte0_all}")
    
    # Analyze pattern
    print("\nPattern analysis:")
    
    # Check if each lane got different bytes
    unique_bytes = set()
    for lane in range(64):
        for b in out_bytes[lane]:
            unique_bytes.add(int(b))
    
    print(f"  Unique byte values across all lanes: {len(unique_bytes)}")
    
    # Try to find the gather formula
    print("\n  Attempting to find gather formula...")
    
    # For lane 0, what addresses were read?
    lane0_bytes = out_bytes[0]
    print(f"  Lane 0 got bytes: {list(lane0_bytes)}")
    
    # Check if it's stride-16 pattern
    if len(lane0_bytes) >= 2:
        stride = lane0_bytes[1] - lane0_bytes[0]
        is_stride_16 = all(lane0_bytes[i+1] - lane0_bytes[i] == 16 for i in range(len(lane0_bytes)-1))
        if is_stride_16:
            print(f"  -> Lane 0: stride-16 pattern starting at {lane0_bytes[0]}")
        else:
            print(f"  -> Lane 0: irregular pattern, differences: {[lane0_bytes[i+1] - lane0_bytes[i] for i in range(len(lane0_bytes)-1)]}")
    
    # Check cross-lane pattern
    print("\n  Cross-lane analysis (byte 0 of each lane):")
    byte0_per_lane = [out_bytes[lane][0] for lane in range(8)]
    print(f"  Lanes 0-7, byte 0: {byte0_per_lane}")
    
    return out_bytes

def main():
    print("=" * 70)
    print("TR_B8 Behavior Experiment")
    print("LDS[i] = i, then read with ds_read_b64_tr_b8")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter9exp_tr_b8E")
    
    # Test various base addresses
    for base_addr in [0, 8, 16, 128, 256]:
        out = run_test(hip, func, base_addr)
        analyze_output(out, base_addr)
    
    hip.hipModuleUnload(module)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Map TR_B8 behavior exactly.
LDS[i] = i % 256, then TR_B8 read - determine read pattern.
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
        '-mcpu=gfx950', '-c', 'exp_tr_b8_map.s', '-o', 'exp_tr_b8_map.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'exp_tr_b8_map.co', 'exp_tr_b8_map.o'], cwd=cwd)
    return str(cwd / 'exp_tr_b8_map.co')

def run_test(hip, func, base_addr):
    out = torch.zeros(64 * 2, dtype=torch.int32, device='cuda')
    args = [ctypes.c_void_p(out.data_ptr()), ctypes.c_uint32(base_addr)]
    args_ptrs = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    return out.cpu().numpy()

def main():
    print("=" * 70)
    print("TR_B8 Read Pattern Mapping")
    print("LDS[i] = i % 256, all lanes use same base_addr")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13exp_tr_b8_mapE")
    
    for base_addr in [0, 8, 64, 128]:
        out = run_test(hip, func, base_addr)
        out_bytes = out.view(np.uint8).reshape(64, 8)
        
        print(f"\n{'='*70}")
        print(f"Base addr = {base_addr}")
        print(f"{'='*70}")
        
        # Derive read formula
        print("\nLane | Bytes[0:7] (what LDS addresses were read?)")
        print("-" * 60)
        
        for lane in [0, 1, 2, 3, 8, 16, 32, 48, 56, 63]:
            # Each byte value = LDS address that was read (since LDS[i] = i % 256)
            addrs = list(out_bytes[lane])
            print(f"{lane:4d} | {addrs}")
        
        # Derive the formula
        print("\nPattern Analysis:")
        
        # Check lane 0
        lane0 = list(out_bytes[0])
        print(f"  Lane 0 reads: {lane0}")
        
        # Check if stride pattern exists within a lane
        if len(set(lane0)) == 1:
            print(f"  -> All 8 bytes are identical: {lane0[0]}")
            print(f"  -> TR_B8 broadcasts LDS[{lane0[0]}] to all 8 bytes")
        else:
            diffs = [lane0[i+1] - lane0[i] for i in range(7)]
            print(f"  -> Differences: {diffs}")
        
        # Cross-lane pattern
        byte0_all = [out_bytes[lane][0] for lane in range(64)]
        print(f"\n  Byte[0] for all lanes: {byte0_all[:16]}... (first 16)")
        
        # Check if it's just (lane % 8) + base
        expected = [(lane % 8) + base_addr for lane in range(16)]
        if byte0_all[:16] == expected:
            print(f"  -> Pattern: byte[0] = base + (lane % 8)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: TR_B8 Behavior")
    print("=" * 70)
    print("""
TR_B8 with same base_addr for all lanes:
- Lane L reads LDS[base + (L % 8)]
- All 8 bytes in a lane are identical (broadcast)
- Lanes 0,8,16,24,32,40,48,56 all read same address
- Lanes 1,9,17,25,33,41,49,57 all read same address
- etc.

This is NOT what we want for MFMA! 
We need each lane to get DIFFERENT bytes for the K dimension.
""")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

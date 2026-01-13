#!/usr/bin/env python3
"""TR_B8 with per-lane addresses: lane uses base = lane * 8"""

import torch
import subprocess
import ctypes
import numpy as np
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'exp_tr_b8_perlane.s', '-o', 'exp_tr_b8_perlane.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'exp_tr_b8_perlane.co', 'exp_tr_b8_perlane.o'], cwd=cwd)
    return str(cwd / 'exp_tr_b8_perlane.co')

def main():
    print("=" * 70)
    print("TR_B8 with Per-Lane Address: base = lane * 8")
    print("LDS[i] = i % 256")
    print("=" * 70)
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter17exp_tr_b8_perlaneE")
    
    out = torch.zeros(64 * 2, dtype=torch.int32, device='cuda')
    args = [ctypes.c_void_p(out.data_ptr())]
    args_ptrs = (ctypes.c_void_p * 1)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 64, 1, 1, 8192, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    out_bytes = out.cpu().numpy().view(np.uint8).reshape(64, 8)
    
    print("\nLane | base_addr | Bytes[0:7]")
    print("-" * 60)
    
    for lane in range(16):
        base = lane * 8
        addrs = list(out_bytes[lane])
        print(f"{lane:4d} |     {base:3d}   | {addrs}")
    
    print("...")
    
    for lane in [32, 48, 63]:
        base = lane * 8
        addrs = list(out_bytes[lane])
        print(f"{lane:4d} |     {base:3d}   | {addrs}")
    
    # Analyze: what did each lane actually read?
    print("\n" + "=" * 70)
    print("ANALYSIS: What LDS addresses did each lane read?")
    print("=" * 70)
    
    # For lane L with base = L*8:
    # - Lane reads LDS[base + (lane % 8)] = LDS[L*8 + (L % 8)]
    # - But with cross-lane transpose...
    
    print("\nLane 0 (base=0):")
    print(f"  Got: {list(out_bytes[0])}")
    print(f"  All 8 bytes are: {out_bytes[0][0]} (broadcast)")
    
    print("\nLane 1 (base=8):")
    print(f"  Got: {list(out_bytes[1])}")
    
    # Check if there's cross-lane sharing
    print("\nCross-lane check:")
    print(f"  Lane 0 byte[0] = {out_bytes[0][0]}")
    print(f"  Lane 8 byte[0] = {out_bytes[8][0]}")
    print(f"  Lane 16 byte[0] = {out_bytes[16][0]}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

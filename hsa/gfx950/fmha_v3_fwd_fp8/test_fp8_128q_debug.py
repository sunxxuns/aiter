#!/usr/bin/env python3
"""Debug 128Q kernel - check intermediate values"""

import torch
import subprocess
import ctypes
import math
from pathlib import Path

def build():
    cwd = Path(__file__).parent
    result = subprocess.run([
        'clang', '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa',
        '-mcpu=gfx950', '-c', 'fwd_fp8_128q.s', '-o', 'fwd_fp8_128q.o'
    ], capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Compile error:\n{result.stderr.decode()}")
        return None
    subprocess.run(['ld.lld', '-shared', '-o', 'fwd_fp8_128q.co', 'fwd_fp8_128q.o'], cwd=cwd)
    return str(cwd / 'fwd_fp8_128q.co')

def reference(Q_fp8, K_fp8, V_fp8):
    scale = 1.0 / math.sqrt(128)
    S = Q_fp8.float() @ K_fp8.float().T * scale
    P = torch.softmax(S, dim=1)
    P_fp8 = P.to(torch.float8_e4m3fn).float()
    return P_fp8 @ V_fp8.float()

def main():
    print("Debug 128Q kernel")
    
    co = build()
    if not co:
        return
    
    _ = torch.zeros(1, device='cuda')
    hip = ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
    
    module = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), co.encode())
    func = ctypes.c_void_p()
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_ZN5aiter13fwd_fp8_128qE")
    
    # Minimal test: V=1, uniform Q and K
    seq_len = 32
    print(f"\nTest: V=1 uniform, seq_len={seq_len}")
    
    Q = torch.ones(128, 128, dtype=torch.float8_e4m3fn, device='cuda')
    K = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda')
    V = torch.ones(seq_len, 128, dtype=torch.float8_e4m3fn, device='cuda')
    O = torch.zeros(128, 128, dtype=torch.float32, device='cuda')
    
    args = [ctypes.c_void_p(x.data_ptr()) for x in [O, Q, K, V]]
    args.append(ctypes.c_uint32(seq_len))
    args_ptrs = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 65536, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    
    print(f"O has NaN: {torch.isnan(O).any().item()}")
    print(f"O has Inf: {torch.isinf(O).any().item()}")
    print(f"O mean: {O.mean().item():.4f}")
    print(f"O[0,0:4]: {O[0,0:4].tolist()}")
    print(f"O[32,0:4]: {O[32,0:4].tolist()}")
    print(f"O[64,0:4]: {O[64,0:4].tolist()}")
    print(f"O[96,0:4]: {O[96,0:4].tolist()}")
    
    # Reference
    O_ref = reference(Q, K, V)
    print(f"\nReference:")
    print(f"O_ref mean: {O_ref.mean().item():.4f}")
    print(f"O_ref[0,0:4]: {O_ref[0,0:4].tolist()}")
    
    # Check intermediate: QK result
    print("\nIntermediate QK check:")
    S = Q.float() @ K.float().T / math.sqrt(128)
    print(f"S (QK scaled) mean: {S.mean().item():.4f}")
    print(f"S[0,0:4]: {S[0,0:4].tolist()}")
    
    P = torch.softmax(S, dim=1)
    print(f"P (softmax) mean: {P.mean().item():.4f}")
    print(f"P sum per row: {P[0].sum().item():.4f}")
    
    hip.hipModuleUnload(module)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dump QK accumulators (v32..v47) for the *second* K-tile-pair iteration (s30==2).

Kernel path:
  - run normally until s30==2
  - after tile1 QK MFMA, if debug_flags 0x20000000 is set, dump v32..v47 and exit

This helps answer: is QK for K rows 64..127 producing spurious nonzeros for Q rows 0..63
under identity-Q/K?
"""

from __future__ import annotations

import ctypes
import os
import torch


def decode_qk(raw: torch.Tensor, rows: int, cols: int, waves_per_block: int = 8) -> torch.Tensor:
    """Decode scaffold QK dump (512 threads x 16 floats) to row-major (rows x cols)."""
    decoded = torch.empty((rows, cols), dtype=torch.float32)
    raw_mat = raw.view(waves_per_block * 64, 16)
    for tid in range(waves_per_block * 64):
        lane = tid & 63
        wave = tid >> 6
        col = lane & 31
        row_base = ((lane >> 5) & 1) * 4
        wave_row = wave * 32
        for i in range(16):
            row = (i % 4) + row_base + (i // 4) * 8 + wave_row
            decoded[row, col] = raw_mat[tid, i]
    return decoded


def main() -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    _ = torch.zeros(1, device="cuda")

    # Use the same shapes as test_scaffold_numerics default (S=64, K_TILES=4)
    B, H, D = 1, 1, 128
    S = int(os.environ.get("DUMP_S", "64"))
    num_q_blocks = (S + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = int(os.environ.get("DUMP_K_TILES", "4"))
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    # Identity Q/K
    Qf32 = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float32)
    Kf32 = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float32)
    for r in range(min(s_k, D, s_q)):
        Qf32[0, 0, r, r] = 1.0
        Kf32[0, 0, r, r] = 1.0
    Q = Qf32.to(torch.float8_e4m3fn)
    K = Kf32.to(torch.float8_e4m3fn)
    V = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)

    # Output: 16 floats per tid (v32..v47) for 512 threads
    out = torch.zeros(512 * 16, device="cuda", dtype=torch.float32)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = out.numel() * 4

    debug_flags = int(os.environ.get("DUMP_DEBUG_FLAGS", "0"), 0) | 0x20000000

    zeros = 0
    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_int32(num_k_tiles),
        ctypes.c_int32(stride_qh),
        ctypes.c_int32(stride_kh),
        ctypes.c_int32(stride_vh),
        ctypes.c_int32(stride_oh),
        ctypes.c_int32(debug_flags),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
        ctypes.c_int32(zeros),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), b"/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co")
    hip.hipModuleGetFunction(ctypes.byref(func), module, b"_fwd_fp8_scaffold")
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    raw = out.detach().cpu()
    decoded = decode_qk(raw, rows=256, cols=32)

    # Expected for identity Q/K and K tiles=4:
    # - This iteration corresponds to K rows 64..95.
    # - decoded[0:64, :] should be ~0
    # - decoded[64:96, :] should be ~I_32
    block0 = decoded[0:64, :32]
    block1 = decoded[64:96, :32]
    print("decoded QK (iter2) block0 [0:8,0:8] (expect ~0):")
    print(block0[:8, :8])
    print(f"absmax decoded[0:64,0:32] = {block0.abs().max().item():.6e}")
    diag = torch.diagonal(block1).cpu()
    off = (block1 - torch.diag(diag)).abs().max().item()
    print("decoded QK (iter2) block1 [0:8,0:8] (expect ~I):")
    print(block1[:8, :8])
    print(f"diag min/max (rows64..95): {diag.min().item():.3f}/{diag.max().item():.3f}")
    print(f"offdiag max (rows64..95): {off:.3f}")


if __name__ == "__main__":
    main()


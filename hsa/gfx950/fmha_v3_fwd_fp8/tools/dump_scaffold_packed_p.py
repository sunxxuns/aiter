#!/usr/bin/env python3
"""
Dump packed P (v48..v55) from fwd_fp8_scaffold.s for identity-P debugging.

This launches the scaffold with:
  - Q/K identity (so P should be identity for the first s_k rows)
  - debug_flags=0x00000800 (dump packed P before any lane-mix and exit)

Output buffer layout (per tid): 8 dwords (32B) = v48..v55.
"""

from __future__ import annotations

import ctypes
import os
from typing import List

import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _unpack_u32_bytes(u: int) -> List[int]:
    return [(u >> (8 * i)) & 0xFF for i in range(4)]


def main() -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    _ = torch.zeros(1, device="cuda")

    # Match numerics default small shape
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

    # V can be zeros; PV won't run because we exit at packed-P dump
    V = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)

    # Output: v48..v55 = 8 dwords per tid
    out = torch.zeros(512 * 8, device="cuda", dtype=torch.uint32)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = out.numel() * 4

    debug_flags = int(os.environ.get("DUMP_DEBUG_FLAGS", "0"), 0) | 0x00000800

    # Remaining v_read knobs (unused for this dump)
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
        ctypes.c_int32(zeros),  # v_read_cb
        ctypes.c_int32(zeros),  # v_read_lane_add
        ctypes.c_int32(zeros),  # v_read_v3_xor
        ctypes.c_int32(zeros),  # v_read_v3_add
        ctypes.c_int32(zeros),  # v_read_v4_add
        ctypes.c_int32(zeros),  # v_read_v2_add
        ctypes.c_int32(zeros),  # v_read_base_add
        ctypes.c_int32(zeros),  # v_read_base_xor
        ctypes.c_int32(zeros),  # v_read_base_extra_add
        ctypes.c_int32(zeros),  # v_read_s25_override
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), CO_PATH.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, KERNEL_NAME)
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    u32 = out.cpu().tolist()

    for tid in (0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 56, 57, 58, 59):
        base = tid * 8
        bs: List[int] = []
        for w in u32[base : base + 8]:
            bs += _unpack_u32_bytes(int(w))
        print(f"tid={tid} lane={tid & 63} v48..v55 bytes[0:32]={bs[:32]}")


if __name__ == "__main__":
    main()


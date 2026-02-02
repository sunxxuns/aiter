#!/usr/bin/env python3
"""
Group-wise TR8 dump: decode dumped bytes into (k, col) pairs.

We use two byte-coded V initializations:
  - Vbytes_row[r,c] = r + 1   (decode k = byte - 1)
  - Vbytes_col[r,c] = c + 1   (decode col = byte - 1)

Then we run the scaffold with:
  - debug_flags = 0x00000001 (enable group dump)
  - v_read_s25_override = group_id (0..3) selects which group dump fires

The kernel dumps [v0..v7, v48..v55] right after the chosen TR8 group's reads and exits.
We read v0..v7 bytes and interpret each byte's (k,col) from the two runs.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Tuple

import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _load_hip() -> ctypes.CDLL:
    return ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")


def _launch(
    hip: ctypes.CDLL,
    *,
    out: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_k_tiles: int,
    stride_qh: int,
    stride_kh: int,
    stride_vh: int,
    stride_oh_bytes: int,
    debug_flags: int,
    v_read_s25_override: int,
) -> None:
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
        ctypes.c_int32(stride_oh_bytes),
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
        ctypes.c_int32(v_read_s25_override),
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


def _u32_to_bytes(words: List[int]) -> List[int]:
    out: List[int] = []
    for w in words:
        out.extend(list(int(w).to_bytes(4, "little")))
    return out


def main() -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")

    # Match numerics small shape: S=64, K_TILES=4 -> s_q=256, s_k=128
    B, H, D = 1, 1, 128
    S = int(os.environ.get("DUMP_S", "64"))
    num_q_blocks = (S + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = int(os.environ.get("DUMP_K_TILES", "4"))
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    # Identity Q/K so the kernel runs the same PV path as identity-P.
    Qf32 = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float32)
    Kf32 = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float32)
    for r in range(min(s_k, D, s_q)):
        Qf32[0, 0, r, r] = 1.0
        Kf32[0, 0, r, r] = 1.0
    Q = Qf32.to(torch.float8_e4m3fn)
    K = Kf32.to(torch.float8_e4m3fn)

    def make_v(pattern: str) -> torch.Tensor:
        v = torch.empty((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)
        vb = v.view(torch.uint8)
        if pattern == "row":
            for r in range(s_k):
                vb[0, 0, r, :] = (r + 1) & 0xFF
        elif pattern == "col":
            for c in range(D):
                vb[0, 0, :, c] = (c + 1) & 0xFF
        else:
            raise ValueError(pattern)
        return v

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D

    tids = [0, 1, 2, 3, 32, 33, 34, 35]
    for group in range(4):
        print(f"\n=== group {group} (dump after group{group}) ===")
        out_row = torch.zeros(512 * 16, device="cuda", dtype=torch.uint32)
        out_col = torch.zeros(512 * 16, device="cuda", dtype=torch.uint32)

        _launch(
            hip,
            out=out_row,
            Q=Q,
            K=K,
            V=make_v("row"),
            num_k_tiles=num_k_tiles,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_oh_bytes=out_row.numel() * 4,
            debug_flags=0x00000001,
            v_read_s25_override=group,
        )
        _launch(
            hip,
            out=out_col,
            Q=Q,
            K=K,
            V=make_v("col"),
            num_k_tiles=num_k_tiles,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_oh_bytes=out_col.numel() * 4,
            debug_flags=0x00000001,
            v_read_s25_override=group,
        )

        row_u32 = out_row.cpu().tolist()
        col_u32 = out_col.cpu().tolist()

        for tid in tids:
            base = tid * 16
            b_row = _u32_to_bytes(row_u32[base : base + 8])  # v0..v7
            b_col = _u32_to_bytes(col_u32[base : base + 8])
            pairs: List[Tuple[int, int]] = []
            for rr, cc in zip(b_row[:32], b_col[:32]):
                k = rr - 1 if rr != 0 else -1
                c = cc - 1 if cc != 0 else -1
                pairs.append((k, c))
            print(f"tid={tid} lane={tid & 63} pairs[0:32]={pairs}")


if __name__ == "__main__":
    main()


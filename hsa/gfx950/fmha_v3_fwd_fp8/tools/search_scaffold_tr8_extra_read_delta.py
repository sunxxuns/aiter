#!/usr/bin/env python3
"""
Brute-force the debug-only extra TR8 read (v24 + delta) to recover a missing k.

Kernel instrumentation (in fwd_fp8_scaffold.s):
  - debug_flags 0x00800000 enables an extra ds_read_b64_tr_b8 into v[232:233]
    using address (v24 + delta_bytes), delta_bytes = ((v_read_cb >> 20) & 0xFF) << 3
  - raw dump (0x01000000) additionally stores v[232:235] at +128B per tid:
      v232, v233, v190 (addr used), v_read_cb

This script:
  - uses the currently best-known base knobs (can be overridden via env vars)
  - scans delta_idx in [0..255] and checks whether tid=32's raw bytes now contain k=57
    either in the original raw TR8 set (v200..v231) or in the extra read bytes (v232..v233).

If it finds a delta that introduces 57, it prints it and exits.
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


def _unpack_u32_bytes(u: int) -> List[int]:
    return [(u >> (8 * i)) & 0xFF for i in range(4)]


def _launch_raw_plus_extra(
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
    v_read_cb: int,
    v_read_lane_add: int,
    v_read_v3_xor: int,
    v_read_v3_add: int,
    v_read_v4_add: int,
    v_read_v2_add: int,
    v_read_base_add: int,
    v_read_base_xor: int,
    v_read_base_extra_add: int,
    v_read_s25_override: int,
) -> None:
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
        ctypes.c_int32(v_read_cb),
        ctypes.c_int32(v_read_lane_add),
        ctypes.c_int32(v_read_v3_xor),
        ctypes.c_int32(v_read_v3_add),
        ctypes.c_int32(v_read_v4_add),
        ctypes.c_int32(v_read_v2_add),
        ctypes.c_int32(v_read_base_add),
        ctypes.c_int32(v_read_base_xor),
        ctypes.c_int32(v_read_base_extra_add),
        ctypes.c_int32(v_read_s25_override),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), CO_PATH.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, KERNEL_NAME)
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 512, 1, 1, 50176, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)


def main() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")

    # Shapes (one block)
    B, H, D = 1, 1, 128
    num_q_blocks = 2
    num_k_tiles = 2
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    Q = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float8_e4m3fn)
    K = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)

    vb = torch.empty((s_k, D), device="cuda", dtype=torch.uint8)
    for k in range(s_k):
        vb[k, :] = k & 0xFF
    V = vb.view(torch.float8_e4m3fn).reshape(B, H, s_k, D)

    # Allocate enough for raw(32 dwords) + extra(4 dwords) = 36 dwords/tid
    out = torch.zeros(512 * 36, device="cuda", dtype=torch.uint32)
    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = out.numel() * 4

    # Base knobs: default to best-known min_cov=31 config
    colblk = int(os.environ.get("X_COLBLK", "0")) & 3
    perm_id = int(os.environ.get("X_PERM_ID", "0"))
    write_mode = int(os.environ.get("X_WRITE_MODE", "0"))
    v23_tw = int(os.environ.get("X_V23_TWEAK", "0")) & 0xF
    v24_tw = int(os.environ.get("X_V24_TWEAK", "0")) & 0xF

    v_read_lane_add = int(os.environ.get("X_LANE_ADD", "0"))
    v_read_v3_xor = int(os.environ.get("X_V3_XOR", "1"))
    v_read_v3_add = int(os.environ.get("X_V3_ADD", "0"))
    v_read_v4_add = int(os.environ.get("X_V4_ADD", "8"))
    v_read_v2_add = int(os.environ.get("X_V2_ADD", "2"))
    v_read_base_add = int(os.environ.get("X_BASE_ADD", "0x40"), 0)
    v_read_base_xor = int(os.environ.get("X_BASE_XOR", "0x0"), 0)
    v_read_base_extra_add = int(os.environ.get("X_BASE_EXTRA_ADD", "0x0"), 0)
    v_read_s25_override = int(os.environ.get("X_S25", "0x0"), 0)

    # flags: raw dump + v_read_cb base path + disable row perm + identity write + extra read
    debug_flags = 0x01000000 | 0x00800000 | 0x00000004 | 0x00000080 | 0x00080000

    tid = 32
    lane = tid & 63
    expect = set(range(32, 64)) if lane >= 32 else set(range(0, 32))

    for base_sel in range(9):
        for delta_idx in range(256):
            v_read_cb = (
                colblk
                | (perm_id << 2)
                | (write_mode << 8)
                | (v23_tw << 12)
                | (v24_tw << 16)
                | ((delta_idx & 0xFF) << 20)
                | ((base_sel & 0xF) << 28)
            )
            out.zero_()
            _launch_raw_plus_extra(
                hip,
                out=out,
                Q=Q,
                K=K,
                V=V,
                num_k_tiles=num_k_tiles,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                stride_oh_bytes=stride_oh,
                debug_flags=debug_flags,
                v_read_cb=v_read_cb,
                v_read_lane_add=v_read_lane_add,
                v_read_v3_xor=v_read_v3_xor,
                v_read_v3_add=v_read_v3_add,
                v_read_v4_add=v_read_v4_add,
                v_read_v2_add=v_read_v2_add,
                v_read_base_add=v_read_base_add,
                v_read_base_xor=v_read_base_xor,
                v_read_base_extra_add=v_read_base_extra_add,
                v_read_s25_override=v_read_s25_override,
            )

            raw = out.cpu().tolist()
            base = tid * 36
            raw_bytes: List[int] = []
            for u in raw[base : base + 32]:
                raw_bytes += _unpack_u32_bytes(int(u))
            extra_bytes: List[int] = []
            for u in raw[base + 32 : base + 34]:
                extra_bytes += _unpack_u32_bytes(int(u))
            addr_used = int(raw[base + 34])

            present = set(b for b in (raw_bytes + extra_bytes) if b <= 63)
            cov = len(present & expect)
            if 57 in present:
                print(
                    f"FOUND base_sel={base_sel} delta_idx={delta_idx} delta_bytes={delta_idx<<7} addr=0x{addr_used:x} "
                    f"cov={cov}/32 extra_bytes={extra_bytes}"
                )
                return

    print("No delta_idx in [0..255] introduced k=57 for tid32.")


if __name__ == "__main__":
    main()


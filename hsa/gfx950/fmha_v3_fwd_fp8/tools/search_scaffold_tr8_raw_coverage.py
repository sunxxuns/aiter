#!/usr/bin/env python3
"""
Deterministic search for PV TR8 *read base* correctness using raw TR8 dumps.

Goal (no statistics):
  For every thread tid (0..511):
    - if lane < 32  : raw TR8 bytes must cover all {0..31}
    - if lane >= 32 : raw TR8 bytes must cover all {32..63}

We use the existing scaffold dump flag:
  - 0x01000000: dump raw TR8 regs v200..v231 (32 dwords = 128B) per tid
and keep:
  - 0x00000004: v_read_cb-driven PV base
  - 0x00000080: disable V row perm

Optionally:
  - 0x00080000: identity V write (RAW_TR8_IDENTITY_WRITE=1)

V is initialized as rowbyte: V[k,*] = k (byte).

Scoring:
  primary: maximize min_coverage across all tids (0..32)
  secondary: minimize max_bad_bytes (bytes outside 0..63) across all tids
"""

from __future__ import annotations

import ctypes
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _load_hip() -> ctypes.CDLL:
    return ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")


def _unpack_u32_bytes(u: int) -> List[int]:
    return [(u >> (8 * i)) & 0xFF for i in range(4)]


@dataclass(frozen=True)
class Cand:
    min_cov: int
    max_bad: int
    v23_tweak: int
    v24_tweak: int
    v24_delta_idx: int
    colblk: int
    perm_id: int
    write_mode: int
    s25: int
    v4_add: int
    base_xor: int
    lane_add: int
    base_add: int
    base_extra_add: int
    v2_add: int
    v3_xor: int
    v3_add: int


def run_once(
    hip: ctypes.CDLL,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    colblk: int,
    v23_tweak: int,
    v24_tweak: int,
    v24_delta_idx: int,
    perm_id: int,
    write_mode: int,
    s25_override: int,
    v4_add: int,
    base_xor: int,
    lane_add: int,
    base_add: int,
    base_extra_add: int,
    v2_add: int,
    v3_xor: int,
    v3_add: int,
) -> Cand:
    _, _, s_q, D = Q.shape
    s_k = K.shape[2]
    num_k_tiles = s_k // 32

    out = torch.zeros(512 * 32, device="cuda", dtype=torch.uint32)
    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = out.numel() * 4

    debug_flags = 0x01000000 | 0x00000004 | 0x00000080
    if os.environ.get("RAW_TR8_IDENTITY_WRITE", "0") == "1":
        debug_flags |= 0x00080000
    v_read_cb = (
        (colblk & 3)
        | (perm_id << 2)
        | (write_mode << 8)
        | ((v23_tweak & 0xF) << 12)
        | ((v24_tweak & 0xF) << 16)
        | ((v24_delta_idx & 0xFF) << 20)
    )

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
        ctypes.c_int32(v_read_cb),
        ctypes.c_int32(lane_add),
        ctypes.c_int32(v3_xor),
        ctypes.c_int32(v3_add),
        ctypes.c_int32(v4_add),
        ctypes.c_int32(v2_add),
        ctypes.c_int32(base_add),
        ctypes.c_int32(base_xor),
        ctypes.c_int32(base_extra_add),
        ctypes.c_int32(s25_override),
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

    raw = out.cpu().tolist()

    min_cov = 32
    max_bad = 0
    for tid in range(512):
        lane = tid & 63
        expect = set(range(0, 32)) if lane < 32 else set(range(32, 64))
        base = tid * 32
        bs: List[int] = []
        for u in raw[base : base + 32]:
            bs += _unpack_u32_bytes(int(u))
        cov = len(set(bs) & expect)
        bad = sum(1 for b in bs if b > 63)
        if cov < min_cov:
            min_cov = cov
        if bad > max_bad:
            max_bad = bad
        if min_cov == 0 and max_bad == 128:
            break

    return Cand(
        min_cov=min_cov,
        max_bad=max_bad,
        v23_tweak=v23_tweak,
        v24_tweak=v24_tweak,
        v24_delta_idx=v24_delta_idx,
        colblk=colblk,
        perm_id=perm_id,
        write_mode=write_mode,
        s25=s25_override,
        v4_add=v4_add,
        base_xor=base_xor,
        lane_add=lane_add,
        base_add=base_add,
        base_extra_add=base_extra_add,
        v2_add=v2_add,
        v3_xor=v3_xor,
        v3_add=v3_add,
    )


def main() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")

    # shape (one block)
    B, H, D = 1, 1, 128
    num_q_blocks = 2
    num_k_tiles = 2
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    Q = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float8_e4m3fn)
    K = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)

    v_bytes = torch.empty((s_k, D), device="cuda", dtype=torch.uint8)
    for r in range(s_k):
        v_bytes[r, :] = r & 0xFF
    V = v_bytes.view(torch.float8_e4m3fn).reshape(B, H, s_k, D)

    rng = random.Random(int(os.environ.get("RAW_TR8_SEED", "0")))
    num_samples = int(os.environ.get("RAW_TR8_SAMPLES", "2000"))

    perm_ids = [0]  # keep identity during search
    write_modes = [0, 2]
    colblks = [0, 1, 2, 3]
    v23_tweaks = list(range(0, 16))
    v24_tweaks = list(range(0, 16))
    v24_delta_idxs = list(range(0, 64))
    s25s = list(range(0, 0x1000, 0x80))
    v4_adds = list(range(0, 64, 8))
    lane_adds = [0, 1]
    v2_adds = [0, 1, 2, 4]
    v3_xors = [0, 1, 2, 4, 8]
    v3_adds = [0, 1, 2, 4, 8]
    base_adds = [0, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0]
    base_extra_adds = [0, 0x400, 0x800, 0x1000]
    xor_bits = [0x20, 0x40, 0x400, 0x800, 0x1000]
    base_xors = [0]
    for b in xor_bits:
        base_xors += [x ^ b for x in list(base_xors)]

    best: List[Cand] = []
    for i in range(1, num_samples + 1):
        c = run_once(
            hip,
            Q,
            K,
            V,
            colblk=rng.choice(colblks),
            v23_tweak=rng.choice(v23_tweaks),
            v24_tweak=rng.choice(v24_tweaks),
            v24_delta_idx=rng.choice(v24_delta_idxs),
            perm_id=rng.choice(perm_ids),
            write_mode=rng.choice(write_modes),
            s25_override=rng.choice(s25s),
            v4_add=rng.choice(v4_adds),
            base_xor=rng.choice(base_xors),
            lane_add=rng.choice(lane_adds),
            base_add=rng.choice(base_adds),
            base_extra_add=rng.choice(base_extra_adds),
            v2_add=rng.choice(v2_adds),
            v3_xor=rng.choice(v3_xors),
            v3_add=rng.choice(v3_adds),
        )
        best.append(c)
        best.sort(key=lambda x: (x.min_cov, -x.max_bad), reverse=True)
        best = best[:10]
        if i % 200 == 0:
            top = best[0]
            print(
                f"[{i:05d}/{num_samples}] min_cov={top.min_cov}/32 max_bad={top.max_bad}/128 "
                f"v23_tw={top.v23_tweak} v24_tw={top.v24_tweak} v24_d={top.v24_delta_idx} colblk={top.colblk} perm={top.perm_id} wm={top.write_mode} s25=0x{top.s25:x} v4_add={top.v4_add} "
                f"v2_add={top.v2_add} v3_xor={top.v3_xor} v3_add={top.v3_add} "
                f"base_add=0x{top.base_add:x} base_xor=0x{top.base_xor:x} base_extra=0x{top.base_extra_add:x} "
                f"lane_add={top.lane_add}"
            )
            if top.min_cov == 32 and top.max_bad == 0:
                break

    print("\nTop candidates:")
    for i, c in enumerate(best):
        print(
            f"[{i}] min_cov={c.min_cov}/32 max_bad={c.max_bad}/128 "
            f"v23_tw={c.v23_tweak} v24_tw={c.v24_tweak} v24_d={c.v24_delta_idx} colblk={c.colblk} perm={c.perm_id} wm={c.write_mode} s25=0x{c.s25:x} v4_add={c.v4_add} "
            f"v2_add={c.v2_add} v3_xor={c.v3_xor} v3_add={c.v3_add} "
            f"base_add=0x{c.base_add:x} base_xor=0x{c.base_xor:x} base_extra=0x{c.base_extra_add:x} "
            f"lane_add={c.lane_add}"
        )


if __name__ == "__main__":
    main()


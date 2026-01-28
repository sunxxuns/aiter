#!/usr/bin/env python3
"""
Random-search PV TR8 (V->A) knobs *in the scaffold* by scoring the packed V->A bytes
after the lane-group selection step.

This is a fast, scaffold-local objective for improving the PV A operand mapping.

What it does:
  - Initializes V as `rowbyte`: each row r stores byte=r in all columns
  - Runs the scaffold kernel in a dump-and-exit mode that dumps the selected packed
    V->A regs (v48..v55) which should represent the 32 FP8 bytes feeding MFMA A.
  - Scores lanes 0 and 32 against expected row-id sets:
      lane0 expects {0..31}
      lane32 expects {32..63}

Notes:
  - We typically force identity V LDS write (debug_flags 0x00080000) and identity
    V row perm (perm_id=0) so the expected sets are meaningful.
  - This does NOT guarantee the final MFMA layout is correct; it’s an objective to
    drive knobs toward a usable mapping, after which we re-run full numerics.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Tuple

import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _load_hip():
    return ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")


def _unpack_u32_bytes(u: int) -> Tuple[int, int, int, int]:
    return (u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF, (u >> 24) & 0xFF)


def _score_bytes(got: list[int], base: int) -> int:
    # Order-free score: how many expected row IDs appear in the 32 packed bytes.
    exp = {(base + k) & 0xFF for k in range(32)}
    return len(set(got[:32]) & exp)


@dataclass(frozen=True)
class Cand:
    score_sum: int
    score0: int
    score32: int
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
    hip,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
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

    # debug dump buffer: 8 dwords per tid (v48..v55)
    out = torch.zeros(512 * 8, device="cuda", dtype=torch.uint32)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = 4 * out.numel()

    # dump selected V->A bytes + use v_read_dump-style knobs + identity write
    debug_flags = 0x02000000 | 0x00000004 | 0x00080000
    v_read_cb = (perm_id << 2) | (write_mode << 8)

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

    def a_bytes_for_lane(lane: int) -> list[int]:
        base = lane * 8
        a_dwords = raw[base : base + 8]  # v48..v55
        bs: list[int] = []
        for u in a_dwords:
            bs.extend(_unpack_u32_bytes(int(u)))
        return bs

    lane0 = a_bytes_for_lane(0)
    lane32b = a_bytes_for_lane(32)
    s0 = _score_bytes(lane0, 0)
    s32 = _score_bytes(lane32b, 32)
    return Cand(
        score_sum=s0 + s32,
        score0=s0,
        score32=s32,
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


def main():
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")  # init context

    # Small fixed shape (one block)
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

    import random

    rng = random.Random(int(os.environ.get("SCAFFOLD_TR8_SEED", "0")))
    num_samples = int(os.environ.get("SCAFFOLD_TR8_SAMPLES", "5000"))

    # For rowbyte scoring, keep perm_id=0 (identity)
    perm_ids = [0]
    write_modes = [0, 2]
    s25s = list(range(0, 0x1000, 0x80))  # 0x000..0xF80
    v4_adds = list(range(0, 64, 8))      # 0..56
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

    best: list[Cand] = []
    for i in range(1, num_samples + 1):
        c = run_once(
            hip,
            Q,
            K,
            V,
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
        best.sort(key=lambda x: (x.score_sum, x.score0, x.score32), reverse=True)
        best = best[:10]
        if i % 200 == 0:
            top = best[0]
            print(
                f"[{i:05d}/{num_samples}] top_sum={top.score_sum} lane0={top.score0} lane32={top.score32} "
                f"perm={top.perm_id} wm={top.write_mode} s25=0x{top.s25:x} v4_add={top.v4_add} "
                f"v2_add={top.v2_add} v3_xor={top.v3_xor} v3_add={top.v3_add} "
                f"base_add=0x{top.base_add:x} base_xor=0x{top.base_xor:x} base_extra=0x{top.base_extra_add:x} "
                f"lane_add={top.lane_add}"
            )

    print("\nTop candidates:")
    for i, c in enumerate(best):
        print(
            f"[{i}] sum={c.score_sum} lane0={c.score0}/32 lane32={c.score32}/32 "
            f"perm={c.perm_id} write_mode={c.write_mode} s25=0x{c.s25:x} "
            f"v4_add={c.v4_add} v2_add={c.v2_add} v3_xor={c.v3_xor} v3_add={c.v3_add} "
            f"base_add=0x{c.base_add:x} base_xor=0x{c.base_xor:x} base_extra=0x{c.base_extra_add:x} "
            f"lane_add={c.lane_add}"
        )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Random-search PV TR8 (V->A) knobs *in the scaffold* by scoring the packed V->A bytes
after the lane-group selection step.

This is a fast, scaffold-local objective for improving the PV A operand mapping.

What it does:
  - Initializes V as `rowbyte`: each row r stores byte=r in all columns
  - Runs the scaffold kernel in a dump-and-exit mode that dumps the selected packed
    V->A regs (v48..v55) which should represent the 32 FP8 bytes feeding MFMA A.
  - Scores lanes 0 and 32 against expected row-id *order* derived from CDNA4 ISA
    MFMA dense layout (FP8 32x32x64):
      - lane0 (thr 0..15 group) consumes k = 0..15 then 32..47
      - lane32 (thr 32..47 group) consumes k = 16..31 then 48..63
    With V initialized to rowbyte (V[k,*]=k), the packed A bytes should match these k
    sequences (positional matches), not just set overlap.

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


def _score_set_overlap(got: list[int], exp_seq: list[int]) -> int:
    """Order-free score: how many expected IDs appear anywhere in the 32 packed bytes."""
    return len(set(got[:32]) & set(exp_seq[:32]))


def _score_positional(got: list[int], exp_seq: list[int]) -> int:
    """Order-aware score: count exact positional matches across the first 32 bytes."""
    return sum(1 for i in range(32) if got[i] == (exp_seq[i] & 0xFF))


@dataclass(frozen=True)
class Cand:
    # Deterministic objective (no statistics):
    # - min_pos: minimum positional matches across all 512 threads (8 waves)
    # - perfect_threads: number of threads with 32/32 positional matches
    # - sum_pos: sum of positional matches across all threads (tie-breaker)
    min_pos: int
    perfect_threads: int
    sum_pos: int
    # A small “spot check” summary for readability
    lane0_pos: int
    lane32_pos: int
    # Which expected ordering variant matched best (see _expected_variants()).
    variant_id: int
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

    # dump selected V->A bytes immediately after V-pack selection (recommended),
    # use v_read_dump-style knobs + identity write, and disable V row perm so
    # rowbyte expectations hold.
    debug_flags = 0x00200000 | 0x00000004 | 0x00080000 | 0x00000080
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

    def a_bytes_for_tid(tid: int) -> list[int]:
        # out layout: 8 dwords per tid (v48..v55)
        base = tid * 8
        a_dwords = raw[base : base + 8]
        bs: list[int] = []
        for u in a_dwords:
            bs.extend(_unpack_u32_bytes(int(u)))
        return bs

    def _expected_variants() -> list[tuple[list[int], list[int]]]:
        """
        Return plausible ISA-consistent expected byte orderings.

        For FP8 32x32x64 (Dense Matrix Layouts) we need, per lane-group:
          lanes 0..31  expect k-set {0..15, 32..47}
          lanes 32..63 expect k-set {16..31, 48..63}

        TR8 notes (DS_READ_B64_TR_B8):
          - one TR8 half brings (0..7,16..23,32..39,48..55)
          - the other half brings the complement

        We don't know the exact packing order in v48..v55 yet, so we score a small
        family of orderings that preserve the correct *k sets* and respect the TR8
        half-split structure. The goal is to find a configuration that is correct
        for every lane under one consistent ordering.
        """
        # group0 (lanes < 32): ks are {0..15,32..47}
        g0_h0 = list(range(0, 8)) + list(range(32, 40))      # TR8 half0 subset
        g0_h1 = list(range(8, 16)) + list(range(40, 48))     # TR8 half1 subset
        # group2 (lanes >= 32): ks are {16..31,48..63}
        g2_h0 = list(range(16, 24)) + list(range(48, 56))
        g2_h1 = list(range(24, 32)) + list(range(56, 64))

        variants: list[tuple[list[int], list[int]]] = []

        # Variant 0: natural k order inside each 16B block
        variants.append((
            list(range(0, 16)) + list(range(32, 48)),
            list(range(16, 32)) + list(range(48, 64)),
        ))

        # Variant 1: TR8 half0 then half1 (each is 16 bytes)
        variants.append((g0_h0 + g0_h1, g2_h0 + g2_h1))

        # Variant 2: swap halves (half1 then half0)
        variants.append((g0_h1 + g0_h0, g2_h1 + g2_h0))

        # Variant 3: interleave in 8B chunks: (half0 first 8B) + (half1 first 8B) + ...
        variants.append((
            list(range(0, 8)) + list(range(8, 16)) + list(range(32, 40)) + list(range(40, 48)),
            list(range(16, 24)) + list(range(24, 32)) + list(range(48, 56)) + list(range(56, 64)),
        ))

        return variants

    variants = _expected_variants()

    # Deterministic “must be correct everywhere” scoring across the whole workgroup:
    # - PV reads of V should not depend on wave id, only lane-group.
    best_tuple = (-1, -1, -1, -1, -1, -1)  # (min_pos, perfect, sum_pos, lane0_pos, lane32_pos, -variant_id)
    best_stats = (0, 0, 0, 0, 0, 0)        # same shape but stores variant_id positively
    for vid, (exp0, exp32) in enumerate(variants):
        min_pos = 32
        perfect = 0
        sum_pos = 0
        for tid in range(512):
            lane = tid & 63
            exp = exp0 if lane < 32 else exp32
            got = a_bytes_for_tid(tid)
            pos = _score_positional(got, exp)
            sum_pos += pos
            if pos < min_pos:
                min_pos = pos
            if pos == 32:
                perfect += 1
        lane0 = a_bytes_for_tid(0)
        lane32b = a_bytes_for_tid(32)
        lane0_pos = _score_positional(lane0, exp0)
        lane32_pos = _score_positional(lane32b, exp32)
        tup = (min_pos, perfect, sum_pos, lane0_pos, lane32_pos, -vid)
        if tup > best_tuple:
            best_tuple = tup
            best_stats = (min_pos, perfect, sum_pos, lane0_pos, lane32_pos, vid)

    min_pos, perfect, sum_pos, lane0_pos, lane32_pos, variant_id = best_stats
    return Cand(
        min_pos=min_pos,
        perfect_threads=perfect,
        sum_pos=sum_pos,
        lane0_pos=lane0_pos,
        lane32_pos=lane32_pos,
        variant_id=variant_id,
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
        best.sort(
            key=lambda x: (
                x.min_pos,
                x.perfect_threads,
                x.sum_pos,
                x.lane0_pos,
                x.lane32_pos,
                -x.variant_id,
            ),
            reverse=True,
        )
        best = best[:10]
        if i % 200 == 0:
            top = best[0]
            print(
                f"[{i:05d}/{num_samples}] min_pos={top.min_pos} perfect={top.perfect_threads}/512 "
                f"lane0_pos={top.lane0_pos} lane32_pos={top.lane32_pos} "
                f"variant={top.variant_id} "
                f"perm={top.perm_id} wm={top.write_mode} s25=0x{top.s25:x} v4_add={top.v4_add} "
                f"v2_add={top.v2_add} v3_xor={top.v3_xor} v3_add={top.v3_add} "
                f"base_add=0x{top.base_add:x} base_xor=0x{top.base_xor:x} base_extra=0x{top.base_extra_add:x} "
                f"lane_add={top.lane_add}"
            )

    print("\nTop candidates:")
    for i, c in enumerate(best):
        print(
            f"[{i}] min_pos={c.min_pos} perfect={c.perfect_threads}/512 sum_pos={c.sum_pos} "
            f"(lane0_pos={c.lane0_pos}/32 lane32_pos={c.lane32_pos}/32) "
            f"variant={c.variant_id} "
            f"perm={c.perm_id} write_mode={c.write_mode} s25=0x{c.s25:x} "
            f"v4_add={c.v4_add} v2_add={c.v2_add} v3_xor={c.v3_xor} v3_add={c.v3_add} "
            f"base_add=0x{c.base_add:x} base_xor=0x{c.base_xor:x} base_extra=0x{c.base_extra_add:x} "
            f"lane_add={c.lane_add}"
        )


if __name__ == "__main__":
    main()


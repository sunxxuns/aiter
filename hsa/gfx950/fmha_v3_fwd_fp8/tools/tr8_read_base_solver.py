#!/usr/bin/env python3
"""
Search for a TR8 read base function that maximizes overlap with the bytes written
by the current preload V write mapping (bitop3:0x78).

We keep the scaffold read structure fixed:
  - bases are base ^ xor_seq[i] for xor_seq = [0,0x20,0x460,0x1020,0x1460,0x60,0x420,0x1060,0x1420]
  - offsets are exactly the scaffold ds_read_b64_tr_b8 sequence (16 reads)

We search a constrained family for base:
  base = base_fn(lane)
  base ^= base_xor
  base ^= ((lane & tid_mask) << tid_shift)
  base ^= ((base & mod_mask) >> mod_shift) << xor_shift   (optional)

Score = number of read *byte* addresses that fall within any written byte.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import csv


def bitop3(a: int, b: int, c: int, ttbl: int) -> int:
    out = 0
    for i in range(32):
        s0 = (a >> i) & 1
        s1 = (b >> i) & 1
        s2 = (c >> i) & 1
        idx = s0 | (s1 << 1) | (s2 << 2)
        bit = (ttbl >> idx) & 1
        out |= (bit << i)
    return out & 0xFFFFFFFF


def tr8_base(lane: int, s25: int) -> int:
    v2 = (lane << 6) & 0xFFFFFFFF
    v3 = (lane << 2) & 0xFFFFFFFF
    v4 = v3 & 48
    v180 = (lane & 3) << 4
    v4 = v4 | v180
    v2 = (v2 & s25) | v4
    v5 = lane & 16
    v6 = (lane << 3) & 0xFFFFFFFF
    v6 = v6 & 8
    v2 = bitop3(v2, v5, v6, 0x36)
    return v2 & 0xFFFFFFFF


def write_base(lane: int) -> int:
    v4 = (lane << 4) & 0xFFFFFFFF
    v4 = bitop3(v4, lane, 0x70, 0x78)
    return v4 & 0xFFFFFFFF


def write_base_bitop3(tid: int, c: int, ttbl: int) -> int:
    v4 = (tid << 4) & 0xFFFFFFFF
    v4 = bitop3(v4, tid, c, ttbl)
    return v4 & 0xFFFFFFFF


def base_linear(lane: int, shifts: Tuple[int, ...]) -> int:
    b = 0
    for sh in shifts:
        b ^= (lane << sh)
    return b & 0xFFFFFFFF


def base_fn(lane: int, kind: str, s25: int, linear_shifts: Tuple[int, ...]) -> int:
    if kind == "tr8":
        return tr8_base(lane, s25)
    if kind == "write":
        return write_base(lane)
    if kind == "linear":
        return base_linear(lane, linear_shifts)
    if kind == "tr8lin":
        return tr8_base(lane, s25) ^ base_linear(lane, linear_shifts)
    raise ValueError(kind)


def apply_xform(
    lane: int,
    kind: str,
    s25: int,
    linear_shifts: Tuple[int, ...],
    base_xor: int,
    tid_mask: int,
    tid_shift: int,
    mod: Tuple[int | None, int | None, int | None],
) -> int:
    b = base_fn(lane, kind, s25, linear_shifts)
    b ^= base_xor
    if tid_mask:
        b ^= ((lane & tid_mask) << tid_shift)
    mod_mask, mod_shift, xor_shift = mod
    if mod_mask is not None:
        b ^= ((b & mod_mask) >> mod_shift) << xor_shift
    return b & 0xFFFFFFFF


XOR_SEQ = [0x0, 0x20, 0x460, 0x1020, 0x1460, 0x60, 0x420, 0x1060, 0x1420]


def scaffold_read_addrs(base0: int, v_lds0: int) -> List[int]:
    bases = [base0 ^ x for x in XOR_SEQ]
    addrs: List[int] = []
    # v20 family (bases[0])
    addrs += [v_lds0 + bases[0] + o for o in (0, 256, 512, 768)]
    # v21..v24 families (bases[1:5])
    addrs += [v_lds0 + bases[1] + 1024]
    addrs += [v_lds0 + bases[2] + 1152]
    addrs += [v_lds0 + bases[3] + 1280]
    addrs += [v_lds0 + bases[4] + 1408]
    # v20 family again
    addrs += [v_lds0 + bases[0] + o for o in (2048, 2176, 2304, 2432)]
    # v25..v28 families (bases[5:9])
    addrs += [v_lds0 + bases[5] + 3072]
    addrs += [v_lds0 + bases[6] + 3200]
    addrs += [v_lds0 + bases[7] + 3328]
    addrs += [v_lds0 + bases[8] + 3456]
    return addrs


def build_write_bytes(v_lds0: int, write_c: int, write_ttbl: int) -> Set[int]:
    w: Set[int] = set()
    for tid in range(256):
        b = v_lds0 + write_base_bitop3(tid, write_c, write_ttbl)  # preload uses tid, not lane
        for i in range(16):
            w.add(b + i)
            w.add(b + 4096 + i)
    return w


def load_scaffold_read_addrs_csv(path: str) -> Set[int]:
    # lane -> [v20..v28]
    vals: Dict[int, List[int]] = {lane: [0] * 9 for lane in range(64)}
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("tid") or not row.get("word") or not row.get("val_u32"):
                continue
            try:
                tid = int(row["tid"])
                lane = tid & 63
                word = int(row["word"])
                val = int(row["val_u32"])
            except ValueError:
                continue
            if 0 <= lane < 64 and 0 <= word <= 8:
                vals[lane][word] = val

    def addrs(v: List[int]) -> List[int]:
        v20, v21, v22, v23, v24, v25, v26, v27, v28 = v
        out: List[int] = []
        out += [v20 + o for o in (0, 256, 512, 768)]
        out += [v21 + 1024, v22 + 1152, v23 + 1280, v24 + 1408]
        out += [v20 + o for o in (2048, 2176, 2304, 2432)]
        out += [v25 + 3072, v26 + 3200, v27 + 3328, v28 + 3456]
        return out

    reads: Set[int] = set()
    for lane in range(64):
        for a in addrs(vals[lane]):
            reads.add(a)
    return reads


@dataclass(frozen=True)
class Cand:
    covered: int
    total: int
    kind: str
    linear_shifts: Tuple[int, ...]
    base_xor: int
    tid_mask: int
    tid_shift: int
    mod: Tuple[int | None, int | None, int | None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    ap.add_argument("--v-lds0", type=int, default=41984)
    ap.add_argument(
        "--csv",
        type=str,
        default="/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/scaffold_tr8_addrs.csv",
        help="CSV dumped from dump_scaffold_tr8_addrs.py",
    )
    ap.add_argument("--write-ttbl", type=lambda x: int(x, 0), default=0x7A)
    ap.add_argument("--write-c", type=lambda x: int(x, 0), default=0x00)
    ap.add_argument(
        "--per-lane-bit",
        action="store_true",
        help="brute force inject a single lane bit at a time (tid_mask one-hot) across many tid_shift values",
    )
    args = ap.parse_args()

    reads = load_scaffold_read_addrs_csv(args.csv)
    write_bytes = build_write_bytes(args.v_lds0, args.write_c, args.write_ttbl)

    # For per-lane-bit brute force, keep the base family simple and focus on injections.
    kinds = ["tr8", "write"] if args.per_lane_bit else ["tr8", "write", "tr8lin", "linear"]
    base_xors = [0x0, 0x20, 0x60, 0x420, 0x460, 0x1020, 0x1060, 0x1420, 0x1460]
    if args.per_lane_bit:
        tid_masks = [0x0] + [(1 << b) for b in range(6)]
        tid_shifts = list(range(0, 13))
    else:
        tid_masks = [0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F]
        tid_shifts = [3, 4, 5, 6]
    mod_candidates = [
        (None, None, None),
        (0x1FF, 7, 3),
        (0x1FF, 7, 4),
        (0x3FF, 8, 3),
        (0x3FF, 8, 4),
        (0x7FF, 9, 3),
        (0x7FF, 9, 4),
        (0xFFF, 10, 3),
        (0xFFF, 10, 4),
    ]
    linear_shift_pool = [2, 3, 4, 5, 6, 7, 8]
    linear_sets = [()]
    if not args.per_lane_bit:
        for k in (1, 2):
            for shs in itertools.combinations(linear_shift_pool, k):
                linear_sets.append(shs)

    start = time.time()
    best: List[Cand] = []
    evals = 0

    for kind in kinds:
        lsets = linear_sets if kind in ("linear", "tr8lin") else [()]
        for linear_shifts in lsets:
            for base_xor in base_xors:
                for tid_mask in tid_masks:
                    for tid_shift in tid_shifts:
                        for mod in mod_candidates:
                            evals += 1
                            if args.time_limit and (time.time() - start) > args.time_limit:
                                break
                            # compute coverage over lanes 0..63, but only count addresses that
                            # are both in the *actual scaffold reads* and in written bytes.
                            covered = 0
                            total = 0
                            for lane in range(64):
                                b0 = apply_xform(
                                    lane, kind, args.s25, linear_shifts,
                                    base_xor, tid_mask, tid_shift, mod
                                )
                                addrs = scaffold_read_addrs(b0, args.v_lds0)
                                total += len(addrs)
                                covered += sum(1 for a in addrs if (a in reads and a in write_bytes))
                            cand = Cand(
                                covered=covered,
                                total=total,
                                kind=kind,
                                linear_shifts=linear_shifts,
                                base_xor=base_xor,
                                tid_mask=tid_mask,
                                tid_shift=tid_shift,
                                mod=mod,
                            )
                            best.append(cand)
                            best.sort(key=lambda c: c.covered, reverse=True)
                            best = best[: args.topk]

    print(f"unique_reads={len(reads)} evals={evals} total_reads={best[0].total if best else 0}")
    print(f"write_map: bitop3(ttbl=0x{args.write_ttbl:02x}, c=0x{args.write_c:02x})")
    for i, c in enumerate(best):
        mm, ms, xs = c.mod
        print(
            f"[{i}] covered={c.covered}/{c.total} kind={c.kind} lin={c.linear_shifts} "
            f"base_xor=0x{c.base_xor:x} tid_mask=0x{c.tid_mask:x} tid_shift={c.tid_shift} "
            f"mod=({mm},{ms},{xs})"
        )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Search for a V LDS write swizzle that maximizes TR8 read coverage.

We search a constrained family of swizzles:
  base = base_fn(tid)           # write base or TR8 base
  base ^= base_xor
  base ^= ((tid & tid_mask) << tid_shift)
  base ^= ((base & mod_mask) >> mod_shift) << xor_shift   # optional

Then write addresses = base + offsets (2 or 4 offsets).
Coverage = #TR8 read addresses hit by any write.
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

from tr8_layout_solver import bitop3, tr8_read_addrs, tr8_base


def write_base(tid: int) -> int:
    v4 = (tid << 4) & 0xFFFFFFFF
    v4 = bitop3(v4, tid, 0x70, 0x78)
    return v4 & 0xFFFFFFFF


def apply_swizzle(
    tid: int,
    base_kind: str,
    base_xor: int,
    tid_mask: int,
    tid_shift: int,
    mod_mask: int | None,
    mod_shift: int | None,
    xor_shift: int | None,
    s25: int,
    linear_shifts: Tuple[int, ...] = (),
    tr8_base_cache: List[int] | None = None,
    write_base_cache: List[int] | None = None,
) -> int:
    if base_kind == "tr8":
        base = tr8_base_cache[tid] if tr8_base_cache else tr8_base(tid, s25)
    elif base_kind == "tr8lin":
        base = tr8_base_cache[tid] if tr8_base_cache else tr8_base(tid, s25)
        for sh in linear_shifts:
            base ^= (tid << sh)
    elif base_kind == "linear":
        base = 0
        for sh in linear_shifts:
            base ^= (tid << sh)
    else:
        base = write_base_cache[tid] if write_base_cache else write_base(tid)
    base ^= base_xor
    if tid_mask:
        base ^= ((tid & tid_mask) << tid_shift)
    if mod_mask is not None and mod_shift is not None and xor_shift is not None:
        base ^= ((base & mod_mask) >> mod_shift) << xor_shift
    return base & 0xFFFFFFFF


def write_addrs_for_lane(
    tid: int,
    base_kind: str,
    base_xor: int,
    tid_mask: int,
    tid_shift: int,
    mod_mask: int | None,
    mod_shift: int | None,
    xor_shift: int | None,
    s25: int,
    base: int,
    offsets: Iterable[int],
    linear_shifts: Tuple[int, ...] = (),
    tr8_base_cache: List[int] | None = None,
    write_base_cache: List[int] | None = None,
) -> Set[int]:
    b = apply_swizzle(
        tid,
        base_kind,
        base_xor,
        tid_mask,
        tid_shift,
        mod_mask,
        mod_shift,
        xor_shift,
        s25,
        linear_shifts,
        tr8_base_cache,
        write_base_cache,
    )
    return {base + b + off for off in offsets}


@dataclass
class Candidate:
    covered: int
    total: int
    base_kind: str
    base_xor: int
    tid_mask: int
    tid_shift: int
    mod_mask: int | None
    mod_shift: int | None
    xor_shift: int | None
    offsets: Tuple[int, ...]
    linear_shifts: Tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400)
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--offsets", type=str, default="0,256,512,768,1024,1152,1280,1408,2048,2176,2304,2432,4096")
    parser.add_argument("--offset-count", type=int, default=2, choices=[2, 4, 6, 8])
    parser.add_argument("--max-offset-combos", type=int, default=0,
                        help="Limit number of offset combinations to search (0 = all)")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=5000,
                        help="Print progress every N evaluations")
    parser.add_argument("--max-evals", type=int, default=0,
                        help="Limit total evaluations (0 = unlimited)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Stop after N seconds (0 = unlimited)")
    args = parser.parse_args()

    base = args.base
    s25 = args.s25
    lanes = args.lanes
    offsets = [int(x, 0) for x in args.offsets.split(",") if x.strip()]

    # Precompute read addresses per lane
    read_addrs_all: List[Set[int]] = []
    for tid in range(lanes):
        read_addrs_all.append(tr8_read_addrs(tid, s25, base))
    total_reads = sum(len(r) for r in read_addrs_all)
    tr8_base_cache = [tr8_base(tid, s25) for tid in range(lanes)]
    write_base_cache = [write_base(tid) for tid in range(lanes)]

    # Search space
    base_kinds = ["write", "tr8", "tr8lin", "linear"]
    base_xors = [0x0, 0x20, 0x60, 0x420, 0x460, 0x1020, 0x1060, 0x1420, 0x1460]
    tid_masks = [0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F]
    tid_shifts = [3, 4, 5, 6, 7]
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

    best: List[Candidate] = []

    combos = itertools.combinations(offsets, args.offset_count)
    if args.max_offset_combos:
        combos = itertools.islice(combos, args.max_offset_combos)

    linear_shift_pool = [2, 3, 4, 5, 6, 7, 8]

    evals = 0
    start = time.time()
    best: List[Candidate] = []

    for offs in combos:
        for base_kind in base_kinds:
            linear_shift_sets = [()]
            if base_kind in ("linear", "tr8lin"):
                linear_shift_sets = []
                for k in (1, 2, 3):
                    for shs in itertools.combinations(linear_shift_pool, k):
                        linear_shift_sets.append(shs)
            for linear_shifts in linear_shift_sets:
                for base_xor in base_xors:
                    for tid_mask in tid_masks:
                        for tid_shift in tid_shifts:
                            for mod_mask, mod_shift, xor_shift in mod_candidates:
                                evals += 1
                                if args.max_evals and evals > args.max_evals:
                                    break
                                if args.time_limit and (time.time() - start) > args.time_limit:
                                    break
                                write_addrs: Set[int] = set()
                                for tid in range(lanes):
                                    write_addrs |= write_addrs_for_lane(
                                        tid,
                                        base_kind,
                                        base_xor,
                                        tid_mask,
                                        tid_shift,
                                        mod_mask,
                                        mod_shift,
                                        xor_shift,
                                        s25,
                                        base,
                                        offs,
                                        linear_shifts,
                                        tr8_base_cache,
                                        write_base_cache,
                                    )
                                covered = 0
                                for reads in read_addrs_all:
                                    covered += sum(1 for a in reads if a in write_addrs)
                                cand = Candidate(
                                    covered=covered,
                                    total=total_reads,
                                    base_kind=base_kind,
                                    base_xor=base_xor,
                                    tid_mask=tid_mask,
                                    tid_shift=tid_shift,
                                    mod_mask=mod_mask,
                                    mod_shift=mod_shift,
                                    xor_shift=xor_shift,
                                    offsets=offs,
                                    linear_shifts=linear_shifts,
                                )
                                best.append(cand)
                                if len(best) > args.topk * 4:
                                    best = sorted(best, key=lambda c: c.covered, reverse=True)[
                                        : args.topk * 2
                                    ]
                                if args.progress_every and evals % args.progress_every == 0:
                                    elapsed = time.time() - start
                                    top = max(best, key=lambda c: c.covered) if best else None
                                    top_cov = top.covered if top else 0
                                    print(
                                        f"[progress] evals={evals} elapsed={elapsed:.1f}s "
                                        f"top_covered={top_cov}"
                                    )
                            if args.time_limit and (time.time() - start) > args.time_limit:
                                break
                            if args.max_evals and evals > args.max_evals:
                                break
                        if args.time_limit and (time.time() - start) > args.time_limit:
                            break
                        if args.max_evals and evals > args.max_evals:
                            break
                    if args.time_limit and (time.time() - start) > args.time_limit:
                        break
                    if args.max_evals and evals > args.max_evals:
                        break
                if args.time_limit and (time.time() - start) > args.time_limit:
                    break
                if args.max_evals and evals > args.max_evals:
                    break
            if args.max_evals and evals > args.max_evals:
                break
            if args.time_limit and (time.time() - start) > args.time_limit:
                break
        if args.max_evals and evals > args.max_evals:
            break
        if args.time_limit and (time.time() - start) > args.time_limit:
            break

        best = sorted(best, key=lambda c: c.covered, reverse=True)[: args.topk * 2]

    best = sorted(best, key=lambda c: c.covered, reverse=True)[: args.topk]
    print(f"total_reads={total_reads}")
    for i, c in enumerate(best):
        print(
            f"[{i}] covered={c.covered} "
            f"base={c.base_kind} base_xor=0x{c.base_xor:x} "
            f"tid_mask=0x{c.tid_mask:x} tid_shift={c.tid_shift} "
            f"mod=({c.mod_mask},{c.mod_shift},{c.xor_shift}) "
            f"offsets={c.offsets} linear_shifts={c.linear_shifts}"
        )


if __name__ == "__main__":
    main()

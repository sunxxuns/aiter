#!/usr/bin/env python3
"""
Search offsets for a fixed write base formula.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Iterable, List, Set, Tuple

from tr8_layout_solver import tr8_read_addrs


def linear_base(tid: int, base_xor: int, tid_mask: int, tid_shift: int, shifts: Iterable[int]) -> int:
    base = 0
    for sh in shifts:
        base ^= (tid << sh)
    if tid_mask:
        base ^= ((tid & tid_mask) << tid_shift)
    base ^= base_xor
    return base & 0xFFFFFFFF


def best_offsets(
    base: int,
    s25: int,
    lanes: int,
    shifts: Iterable[int],
    base_xor: int,
    tid_mask: int,
    tid_shift: int,
    offsets: List[int],
    offset_count: int,
    max_combos: int,
) -> Tuple[int, Tuple[int, ...]]:
    reads_all = [tr8_read_addrs(tid, s25, base) for tid in range(lanes)]
    total_reads = sum(len(r) for r in reads_all)

    combos = itertools.combinations(offsets, offset_count)
    if max_combos:
        combos = itertools.islice(combos, max_combos)

    best = (0, ())
    for offs in combos:
        writes: Set[int] = set()
        for tid in range(lanes):
            b = linear_base(tid, base_xor, tid_mask, tid_shift, shifts)
            for off in offs:
                writes.add(base + b + off)
        covered = 0
        for reads in reads_all:
            covered += sum(1 for a in reads if a in writes)
        if covered > best[0]:
            best = (covered, offs)
    return total_reads, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400)
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--offsets", type=str, default="0,256,512,768,1024,1152,1280,1408,2048,2176,2304,2432,4096")
    parser.add_argument("--offset-count", type=int, default=6, choices=[4, 6, 8])
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--shifts", type=str, default="3,7")
    parser.add_argument("--base-xor", type=lambda x: int(x, 0), default=0x20)
    parser.add_argument("--tid-mask", type=lambda x: int(x, 0), default=0x3)
    parser.add_argument("--tid-shift", type=int, default=7)
    args = parser.parse_args()

    offsets = [int(x, 0) for x in args.offsets.split(",") if x.strip()]
    shifts = [int(x) for x in args.shifts.split(",") if x.strip()]

    total_reads, best = best_offsets(
        args.base,
        args.s25,
        args.lanes,
        shifts,
        args.base_xor,
        args.tid_mask,
        args.tid_shift,
        offsets,
        args.offset_count,
        args.max_combos,
    )

    print(f"total_reads={total_reads}")
    print(f"best covered={best[0]} offsets={best[1]}")


if __name__ == "__main__":
    main()

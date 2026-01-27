#!/usr/bin/env python3
"""
Search two-base write layouts to increase TR8 read coverage.
Each candidate uses two base functions and a shared offset set.
"""
from __future__ import annotations

import argparse
import itertools
import time
from typing import Iterable, List, Set, Tuple

from tr8_layout_solver import bitop3, tr8_read_addrs, tr8_base


def write_base(tid: int) -> int:
    v4 = (tid << 4) & 0xFFFFFFFF
    v4 = bitop3(v4, tid, 0x70, 0x78)
    return v4 & 0xFFFFFFFF


def base_linear(tid: int, shifts: Tuple[int, ...]) -> int:
    base = 0
    for sh in shifts:
        base ^= (tid << sh)
    return base & 0xFFFFFFFF


def base_fn(tid: int, kind: str, shifts: Tuple[int, ...], s25: int) -> int:
    if kind == "tr8":
        return tr8_base(tid, s25)
    if kind == "write":
        return write_base(tid)
    if kind == "linear":
        return base_linear(tid, shifts)
    if kind == "tr8lin":
        return tr8_base(tid, s25) ^ base_linear(tid, shifts)
    return 0


def write_addrs_for_lane(
    tid: int,
    kind0: str,
    kind1: str,
    shifts0: Tuple[int, ...],
    shifts1: Tuple[int, ...],
    base_xor0: int,
    base_xor1: int,
    s25: int,
    base: int,
    offsets: Iterable[int],
) -> Set[int]:
    b0 = base_fn(tid, kind0, shifts0, s25) ^ base_xor0
    b1 = base_fn(tid, kind1, shifts1, s25) ^ base_xor1
    addrs = set()
    for off in offsets:
        addrs.add(base + b0 + off)
        addrs.add(base + b1 + off)
    return addrs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400)
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--offsets", type=str, default="0,256,512,768,1024,1152,1280,1408")
    parser.add_argument("--offset-count", type=int, default=6, choices=[4, 6, 8, 10, 12])
    parser.add_argument("--max-offset-combos", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=20000)
    parser.add_argument("--time-limit", type=int, default=60)
    args = parser.parse_args()

    base = args.base
    s25 = args.s25
    lanes = args.lanes
    offsets = [int(x, 0) for x in args.offsets.split(",") if x.strip()]

    read_addrs_all: List[Set[int]] = []
    for tid in range(lanes):
        read_addrs_all.append(tr8_read_addrs(tid, s25, base))
    total_reads = sum(len(r) for r in read_addrs_all)

    kinds = ["tr8", "write", "linear", "tr8lin"]
    base_xors = [0x0, 0x20, 0x60, 0x420, 0x460, 0x1020, 0x1060, 0x1420, 0x1460]
    linear_shift_pool = [2, 3, 4, 5, 6, 7, 8]
    linear_sets = [()]
    for k in (1, 2):
        for shs in itertools.combinations(linear_shift_pool, k):
            linear_sets.append(shs)

    combos = itertools.combinations(offsets, args.offset_count)
    if args.max_offset_combos:
        combos = itertools.islice(combos, args.max_offset_combos)

    start = time.time()
    evals = 0
    best: List[tuple[int, tuple]] = []

    for offs in combos:
        for kind0 in kinds:
            for kind1 in kinds:
                for shifts0 in (linear_sets if kind0 in ("linear", "tr8lin") else [()]):
                    for shifts1 in (linear_sets if kind1 in ("linear", "tr8lin") else [()]):
                        for bx0 in base_xors:
                            for bx1 in base_xors:
                                evals += 1
                                if args.time_limit and (time.time() - start) > args.time_limit:
                                    break
                                write_addrs: Set[int] = set()
                                for tid in range(lanes):
                                    write_addrs |= write_addrs_for_lane(
                                        tid, kind0, kind1, shifts0, shifts1, bx0, bx1, s25, base, offs
                                    )
                                covered = 0
                                for reads in read_addrs_all:
                                    covered += sum(1 for a in reads if a in write_addrs)
                                best.append((covered, (kind0, kind1, shifts0, shifts1, bx0, bx1, offs)))
                                if len(best) > args.topk * 6:
                                    best = sorted(best, key=lambda x: x[0], reverse=True)[: args.topk * 3]
                                if args.progress_every and evals % args.progress_every == 0:
                                    top = max(best, key=lambda x: x[0])[0] if best else 0
                                    elapsed = time.time() - start
                                    print(f"[progress] evals={evals} elapsed={elapsed:.1f}s top_covered={top}")
                            if args.time_limit and (time.time() - start) > args.time_limit:
                                break
                        if args.time_limit and (time.time() - start) > args.time_limit:
                            break
                    if args.time_limit and (time.time() - start) > args.time_limit:
                        break
                if args.time_limit and (time.time() - start) > args.time_limit:
                    break
            if args.time_limit and (time.time() - start) > args.time_limit:
                break
        if args.time_limit and (time.time() - start) > args.time_limit:
            break

    best = sorted(best, key=lambda x: x[0], reverse=True)[: args.topk]
    print(f"total_reads={total_reads}")
    for i, (cov, cfg) in enumerate(best):
        kind0, kind1, shifts0, shifts1, bx0, bx1, offs = cfg
        print(
            f"[{i}] covered={cov} k0={kind0} k1={kind1} "
            f"sh0={shifts0} sh1={shifts1} bx0=0x{bx0:x} bx1=0x{bx1:x} offs={offs}"
        )


if __name__ == "__main__":
    main()


def write_addrs_for_lane(
    tid: int,
    kind0: str,
    kind1: str,
    shifts0: Tuple[int, ...],
    shifts1: Tuple[int, ...],
    base_xor0: int,
    base_xor1: int,
    s25: int,
    base: int,
    offsets: Iterable[int],
) -> Set[int]:
    b0 = base_fn(tid, kind0, shifts0, s25) ^ base_xor0
    b1 = base_fn(tid, kind1, shifts1, s25) ^ base_xor1
    addrs = set()
    for off in offsets:
        addrs.add(base + b0 + off)
        addrs.add(base + b1 + off)
    return addrs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400)
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--offsets", type=str, default="0,256,512,768,1024,1152,1280,1408")
    parser.add_argument("--offset-count", type=int, default=6, choices=[4, 6, 8, 10, 12])
    parser.add_argument("--max-offset-combos", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=20000)
    parser.add_argument("--time-limit", type=int, default=60)
    args = parser.parse_args()

    base = args.base
    s25 = args.s25
    lanes = args.lanes
    offsets = [int(x, 0) for x in args.offsets.split(",") if x.strip()]

    read_addrs_all: List[Set[int]] = []
    for tid in range(lanes):
        read_addrs_all.append(tr8_read_addrs(tid, s25, base))
    total_reads = sum(len(r) for r in read_addrs_all)

    kinds = ["tr8", "write", "linear", "tr8lin"]
    base_xors = [0x0, 0x20, 0x60, 0x420, 0x460, 0x1020, 0x1060, 0x1420, 0x1460]
    linear_shift_pool = [2, 3, 4, 5, 6, 7, 8]
    linear_sets = [()]
    for k in (1, 2):
        for shs in itertools.combinations(linear_shift_pool, k):
            linear_sets.append(shs)

    combos = itertools.combinations(offsets, args.offset_count)
    if args.max_offset_combos:
        combos = itertools.islice(combos, args.max_offset_combos)

    start = time.time()
    evals = 0
    best: List[tuple[int, tuple]] = []

    for offs in combos:
        for kind0 in kinds:
            for kind1 in kinds:
                for shifts0 in (linear_sets if kind0 in ("linear", "tr8lin") else [()]):
                    for shifts1 in (linear_sets if kind1 in ("linear", "tr8lin") else [()]):
                        for bx0 in base_xors:
                            for bx1 in base_xors:
                                evals += 1
                                if args.time_limit and (time.time() - start) > args.time_limit:
                                    break
                                write_addrs: Set[int] = set()
                                for tid in range(lanes):
                                    write_addrs |= write_addrs_for_lane(
                                        tid, kind0, kind1, shifts0, shifts1, bx0, bx1, s25, base, offs
                                    )
                                covered = 0
                                for reads in read_addrs_all:
                                    covered += sum(1 for a in reads if a in write_addrs)
                                best.append((covered, (kind0, kind1, shifts0, shifts1, bx0, bx1, offs)))
                                if len(best) > args.topk * 6:
                                    best = sorted(best, key=lambda x: x[0], reverse=True)[: args.topk * 3]
                                if args.progress_every and evals % args.progress_every == 0:
                                    top = max(best, key=lambda x: x[0])[0] if best else 0
                                    elapsed = time.time() - start
                                    print(f"[progress] evals={evals} elapsed={elapsed:.1f}s top_covered={top}")
                            if args.time_limit and (time.time() - start) > args.time_limit:
                                break
                        if args.time_limit and (time.time() - start) > args.time_limit:
                            break
                    if args.time_limit and (time.time() - start) > args.time_limit:
                        break
                if args.time_limit and (time.time() - start) > args.time_limit:
                    break
            if args.time_limit and (time.time() - start) > args.time_limit:
                break
        if args.time_limit and (time.time() - start) > args.time_limit:
            break

    best = sorted(best, key=lambda x: x[0], reverse=True)[: args.topk]
    print(f"total_reads={total_reads}")
    for i, (cov, cfg) in enumerate(best):
        kind0, kind1, shifts0, shifts1, bx0, bx1, offs = cfg
        print(
            f"[{i}] covered={cov} k0={kind0} k1={kind1} "
            f"sh0={shifts0} sh1={shifts1} bx0=0x{bx0:x} bx1=0x{bx1:x} offs={offs}"
        )


if __name__ == "__main__":
    main()

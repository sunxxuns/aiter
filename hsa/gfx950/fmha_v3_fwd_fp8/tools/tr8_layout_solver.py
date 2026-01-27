#!/usr/bin/env python3
"""
TR8 layout solver: compare TR8 read addresses vs V write addresses.

Default formulas mirror current scaffold:
  - TR8 base: bitop3 0x36 on v2, v5, v6 with s25 mask and xors
  - V write: bitop3 0x78 on (tid<<4, tid, 0x70)

Usage examples:
  python tools/tr8_layout_solver.py --lanes 64
  python tools/tr8_layout_solver.py --s25 0xb80 --write-offsets 0,4096
"""
from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from typing import Iterable, List, Set


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


def tr8_base(tid: int, s25: int) -> int:
    v2 = (tid << 6) & 0xFFFFFFFF
    v3 = (tid << 2) & 0xFFFFFFFF
    v4 = v3 & 48
    v180 = (tid & 3) << 4
    v4 = v4 | v180
    v2 = (v2 & s25) | v4
    v5 = tid & 16
    v6 = (tid << 3) & 0xFFFFFFFF
    v6 = v6 & 8
    v2 = bitop3(v2, v5, v6, 0x36)
    return v2 & 0xFFFFFFFF


def tr8_bases(tid: int, s25: int) -> List[int]:
    v2 = tr8_base(tid, s25)
    xor_seq = [0x0, 0x20, 0x460, 0x1020, 0x1460, 0x60, 0x420, 0x1060, 0x1420]
    return [v2 ^ x for x in xor_seq]


def tr8_read_addrs(tid: int, s25: int, base: int) -> Set[int]:
    bases = tr8_bases(tid, s25)
    addrs: Set[int] = set()
    # set0: base0 offsets 0,256,512,768
    for off in (0, 256, 512, 768):
        addrs.add(base + bases[0] + off)
    # set1: base1..4 offsets 1024,1152,1280,1408
    for b, off in zip(bases[1:5], (1024, 1152, 1280, 1408)):
        addrs.add(base + b + off)
    # set2: base5..8 offsets 2048,2176,2304,2432
    for b, off in zip(bases[5:9], (2048, 2176, 2304, 2432)):
        addrs.add(base + b + off)
    return addrs


def v_write_addr(tid: int, base: int) -> int:
    v4 = (tid << 4) & 0xFFFFFFFF
    v4 = bitop3(v4, tid, 0x70, 0x78)
    return (base + v4) & 0xFFFFFFFF


def v_write_addr_swizzled(
    tid: int,
    base: int,
    mod_mask: int | None = None,
    mod_shift: int | None = None,
    xor_shift: int | None = None,
    use_tr8_base: bool = False,
    s25: int = 0xB80,
    base_xor: int = 0,
) -> int:
    if use_tr8_base:
        v4 = tr8_base(tid, s25)
    else:
        v4 = (tid << 4) & 0xFFFFFFFF
        v4 = bitop3(v4, tid, 0x70, 0x78)
    if mod_mask is not None and mod_shift is not None and xor_shift is not None:
        v4 ^= ((v4 & mod_mask) >> mod_shift) << xor_shift
    v4 ^= base_xor
    return (base + v4) & 0xFFFFFFFF


def v_write_addrs(
    tid: int,
    base: int,
    write_offsets: Iterable[int],
    mod_mask: int | None = None,
    mod_shift: int | None = None,
    xor_shift: int | None = None,
    use_tr8_base: bool = False,
    s25: int = 0xB80,
    base_xor: int = 0,
) -> Set[int]:
    addr = v_write_addr_swizzled(
        tid, base, mod_mask, mod_shift, xor_shift, use_tr8_base, s25, base_xor
    )
    return {addr + off for off in write_offsets}


@dataclass
class Coverage:
    lane: int
    total_reads: int
    covered_reads: int
    missing_reads: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400,
                        help="V LDS base (default 0xA400 = 41984)")
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80,
                        help="s25 mask for TR8 base (default 0xb80)")
    parser.add_argument("--lanes", type=int, default=64, help="Number of lanes")
    parser.add_argument("--write-offsets", type=str, default="0,4096",
                        help="Comma-separated write offsets from base")
    parser.add_argument("--from-dump", type=str, default="",
                        help="CSV from NUMERICS_RAW_DUMP=3 with v20..v28 base addrs")
    parser.add_argument("--search-swizzle", action="store_true",
                        help="Search XOR swizzle variants for best coverage")
    parser.add_argument("--search-offsets", action="store_true",
                        help="Search write offsets for best coverage")
    parser.add_argument("--write-tr8-base", action="store_true",
                        help="Use TR8 base (bitop3 0x36) for write address")
    parser.add_argument("--search-base-xor", action="store_true",
                        help="Search write base XOR constants for best coverage")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only global coverage summary")
    args = parser.parse_args()

    write_offsets = [int(x, 0) for x in args.write_offsets.split(",") if x.strip()]
    base = args.base
    s25 = args.s25

    all_write_addrs: Set[int] = set()
    for tid in range(args.lanes):
        all_write_addrs |= v_write_addrs(
            tid, base, write_offsets, use_tr8_base=args.write_tr8_base, s25=s25
        )

    coverages: List[Coverage] = []
    total_reads = 0
    total_covered = 0
    for tid in range(args.lanes):
        reads = tr8_read_addrs(tid, s25, base)
        covered = sum(1 for a in reads if a in all_write_addrs)
        coverages.append(
            Coverage(tid, len(reads), covered, len(reads) - covered)
        )
        total_reads += len(reads)
        total_covered += covered

    if not args.summary_only:
        for c in coverages:
            print(f"lane {c.lane:02d}: reads={c.total_reads} "
                  f"covered={c.covered_reads} missing={c.missing_reads}")

    print("\nSummary:")
    print(f"  lanes={args.lanes} base=0x{base:x} s25=0x{s25:x} write_offsets={write_offsets}")
    print(f"  total_reads={total_reads} covered={total_covered} missing={total_reads - total_covered}")

    if args.from_dump:
        with open(args.from_dump, "r") as f:
            rows = list(csv.DictReader(f))
        vals = [int(r["val_u32"]) for r in rows]
        if len(vals) < 10:
            raise RuntimeError("dump CSV must include v20..v28 and tid (10 words)")
        v20_27 = vals[:8]
        v28 = vals[8]
        read_addrs = set()
        for off in (0, 256, 512, 768):
            read_addrs.add(v20_27[0] + off)
        for b, off in zip(v20_27[1:5], (1024, 1152, 1280, 1408)):
            read_addrs.add(b + off)
        for b, off in zip(v20_27[5:8] + [v28], (2048, 2176, 2304, 2432)):
            read_addrs.add(b + off)
        covered = sum(1 for a in read_addrs if a in all_write_addrs)
        print("\nLane0 dump coverage:")
        print(f"  read_addrs={len(read_addrs)} covered={covered} missing={len(read_addrs) - covered}")

        if args.search_swizzle:
            candidates = [
                (0x1FF, 7, 3),
                (0x1FF, 7, 4),
                (0x3FF, 8, 3),
                (0x3FF, 8, 4),
                (0x7FF, 9, 3),
                (0x7FF, 9, 4),
            ]
            best = (covered, None)
            for mod_mask, mod_shift, xor_shift in candidates:
                write_addrs = set()
                for tid in range(args.lanes):
                    write_addrs |= v_write_addrs(
                        tid,
                        base,
                        write_offsets,
                        mod_mask,
                        mod_shift,
                        xor_shift,
                        args.write_tr8_base,
                        s25,
                    )
                cov = sum(1 for a in read_addrs if a in write_addrs)
                if cov > best[0]:
                    best = (cov, (mod_mask, mod_shift, xor_shift))
            print("\nSwizzle search best:")
            print(f"  covered={best[0]} candidate={best[1]}")

        if args.search_offsets:
            candidate_offsets = [
                0, 256, 512, 768, 1024, 1152, 1280, 1408, 2048, 2176, 2304, 2432, 4096
            ]
            best = (covered, None)
            for comb in itertools.combinations(candidate_offsets, 2):
                write_addrs = set()
                for tid in range(args.lanes):
                    write_addrs |= v_write_addrs(
                        tid,
                        base,
                        comb,
                        use_tr8_base=args.write_tr8_base,
                        s25=s25,
                    )
                cov = sum(1 for a in read_addrs if a in write_addrs)
                if cov > best[0]:
                    best = (cov, comb)
            print("\nOffset search best:")
            print(f"  covered={best[0]} offsets={best[1]}")

            best4 = (covered, None)
            for comb in itertools.combinations(candidate_offsets, 4):
                write_addrs = set()
                for tid in range(args.lanes):
                    write_addrs |= v_write_addrs(
                        tid,
                        base,
                        comb,
                        use_tr8_base=args.write_tr8_base,
                        s25=s25,
                    )
                cov = sum(1 for a in read_addrs if a in write_addrs)
                if cov > best4[0]:
                    best4 = (cov, comb)
            print("\nOffset search (4 writes) best:")
            print(f"  covered={best4[0]} offsets={best4[1]}")

        if args.search_base_xor:
            candidates = [0x0, 0x20, 0x60, 0x420, 0x460, 0x1020, 0x1060, 0x1420, 0x1460]
            best = (covered, None)
            for bx in candidates:
                write_addrs = set()
                for tid in range(args.lanes):
                    write_addrs |= v_write_addrs(
                        tid,
                        base,
                        write_offsets,
                        use_tr8_base=args.write_tr8_base,
                        s25=s25,
                        base_xor=bx,
                    )
                cov = sum(1 for a in read_addrs if a in write_addrs)
                if cov > best[0]:
                    best = (cov, bx)
            print("\nBase XOR search best:")
            print(f"  covered={best[0]} base_xor=0x{best[1]:x}" if best[1] is not None else "  no improvement")


if __name__ == "__main__":
    main()

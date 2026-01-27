#!/usr/bin/env python3
"""
Fit an affine XOR model for a write base from TR8 read addresses.

Model per bit:
  base_bit = c ^ (a0*t0) ^ (a1*t1) ^ ... ^ (a5*t5)

We derive a "best" base per lane by choosing the base that covers
the most read addrs for a given offset set, then fit a model per bit.
"""
from __future__ import annotations

import argparse
from typing import Iterable, List, Set, Tuple

from tr8_layout_solver import tr8_read_addrs


def best_base_for_lane(reads: Set[int], offsets: Iterable[int], base: int) -> int:
    # Choose base that maximizes hits among read addresses
    counts = {}
    for addr in reads:
        for off in offsets:
            b = addr - base - off
            counts[b] = counts.get(b, 0) + 1
    # pick base with max count
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return best & 0xFFFFFFFF


def best_base_count(reads: Set[int], offsets: Iterable[int], base: int) -> int:
    counts = {}
    for addr in reads:
        for off in offsets:
            b = addr - base - off
            counts[b] = counts.get(b, 0) + 1
    return max(counts.values()) if counts else 0


def fit_affine_bit(bases: List[int], lanes: int, bit: int, vars_count: int) -> Tuple[int, int, int]:
    # brute force all 2^(vars_count+1) affine functions
    best_acc = -1
    best_mask = 0
    best_c = 0
    for mask in range(1 << vars_count):
        for c in (0, 1):
            acc = 0
            for tid in range(lanes):
                val = c
                for i in range(vars_count):
                    if (mask >> i) & 1:
                        val ^= (tid >> i) & 1
                bit_val = (bases[tid] >> bit) & 1
                if val == bit_val:
                    acc += 1
            if acc > best_acc:
                best_acc = acc
                best_mask = mask
                best_c = c
    return best_acc, best_mask, best_c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400)
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--offsets", type=str, default="0,256,768,1024")
    parser.add_argument("--bits", type=int, default=12)
    parser.add_argument("--vars", type=int, default=6,
                        help="Number of tid bits to use in affine fit (default 6)")
    parser.add_argument("--mask", type=lambda x: int(x, 0), default=0,
                        help="Optional bitmask of tid bits to use (overrides --vars)")
    args = parser.parse_args()

    base = args.base
    s25 = args.s25
    lanes = args.lanes
    offsets = [int(x, 0) for x in args.offsets.split(",") if x.strip()]

    bases: List[int] = []
    counts: List[int] = []
    for tid in range(lanes):
        reads = tr8_read_addrs(tid, s25, base)
        b = best_base_for_lane(reads, offsets, base)
        counts.append(best_base_count(reads, offsets, base))
        bases.append(b)

    print("Per-lane best base hit counts:")
    print(f"  min={min(counts)} max={max(counts)} avg={sum(counts)/len(counts):.2f}")

    print("Per-bit affine fit (accuracy out of lanes):")
    if args.mask:
        # remap tid bits by mask
        tid_bits = [i for i in range(6) if (args.mask >> i) & 1]
        vars_count = len(tid_bits)
        def fit_bit_masked(bit: int) -> tuple[int, int, int]:
            best_acc = -1
            best_mask = 0
            best_c = 0
            for mask in range(1 << vars_count):
                for c in (0, 1):
                    acc = 0
                    for tid in range(lanes):
                        val = c
                        for idx, tbit in enumerate(tid_bits):
                            if (mask >> idx) & 1:
                                val ^= (tid >> tbit) & 1
                        bit_val = (bases[tid] >> bit) & 1
                        if val == bit_val:
                            acc += 1
                    if acc > best_acc:
                        best_acc = acc
                        best_mask = mask
                        best_c = c
            return best_acc, best_mask, best_c

        for bit in range(args.bits):
            acc, mask, c = fit_bit_masked(bit)
            print(f"bit{bit:02d}: acc={acc}/{lanes} mask=0x{mask:x} c={c}")
    else:
        for bit in range(args.bits):
            acc, mask, c = fit_affine_bit(bases, lanes, bit, args.vars)
            print(f"bit{bit:02d}: acc={acc}/{lanes} mask=0x{mask:x} c={c}")


if __name__ == "__main__":
    main()

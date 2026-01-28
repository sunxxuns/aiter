#!/usr/bin/env python3
"""
Brute-force bitop3 truth tables for the V write swizzle.

We search write_base(tid) = bitop3(tid<<4, tid, C, TTBL), with C and TTBL varied.

Constraints:
  - lanes_write = 256
  - write_base in [0,4096) and 16B-aligned
  - bases form a permutation (unique across tids)

Objective:
  maximize overlap between scaffold TR8 read byte addresses and written bytes.

This uses scaffold_tr8_addrs.csv (dump_scaffold_tr8_addrs.py) for the exact read bases.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


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


def load_scaffold_read_addrs(path: str) -> Set[int]:
    # lane -> v20..v28 (word 0..8)
    bases: Dict[int, List[int]] = {lane: [0] * 9 for lane in range(64)}
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = int(row["tid"])
            lane = tid & 63
            word = int(row["word"])
            if word <= 8:
                bases[lane][word] = int(row["val_u32"])

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
        for a in addrs(bases[lane]):
            reads.add(a)
    return reads


@dataclass(frozen=True)
class Cand:
    covered: int
    reads: int
    ttbl: int
    c: int


def main() -> None:
    reads = load_scaffold_read_addrs("/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/scaffold_tr8_addrs.csv")

    # Try a few C constants (s26 is 0x70 today, but we allow a bit more)
    c_candidates = [0x00, 0x0F, 0x30, 0x70, 0xFF]

    best: List[Cand] = []

    for c in c_candidates:
        for ttbl in range(256):
            bases = []
            ok = True
            for tid in range(256):
                a = (tid << 4) & 0xFFFFFFFF
                b = tid & 0xFFFFFFFF
                base = bitop3(a, b, c, ttbl) & 0xFFFFFFFF
                if (base & 0xF) != 0 or base < 0 or base >= 4096:
                    ok = False
                    break
                bases.append(base)
            if not ok:
                continue
            if len(set(bases)) != 256:
                continue

            write_bytes: Set[int] = set()
            V_LDS0 = 41984
            for tid, base in enumerate(bases):
                baddr = V_LDS0 + base
                for i in range(16):
                    write_bytes.add(baddr + i)
                    write_bytes.add(baddr + 4096 + i)

            covered = sum(1 for a in reads if a in write_bytes)
            cand = Cand(covered=covered, reads=len(reads), ttbl=ttbl, c=c)
            best.append(cand)
            best.sort(key=lambda x: x.covered, reverse=True)
            best = best[:10]

    print(f"unique_reads={best[0].reads if best else 0}")
    for i, c in enumerate(best):
        print(f"[{i}] covered={c.covered}/{c.reads} ttbl=0x{c.ttbl:02x} c=0x{c.c:02x}")


if __name__ == "__main__":
    main()


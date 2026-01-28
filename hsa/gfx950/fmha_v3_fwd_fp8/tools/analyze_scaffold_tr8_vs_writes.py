#!/usr/bin/env python3
"""
Analyze how many scaffold TR8 read byte addresses are actually covered by the
current V LDS write swizzle (bitop3:0x78) for the preload stage.

This uses `scaffold_tr8_addrs.csv` produced by dump_scaffold_tr8_addrs.py.
"""

from __future__ import annotations

import csv
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


def write_addr_bitop3(tid: int, c: int, ttbl: int) -> int:
    v4 = (tid << 4) & 0xFFFFFFFF
    v4 = bitop3(v4, tid, c, ttbl)
    return v4 & 0xFFFFFFFF


def load_scaffold_bases(path: str) -> Dict[int, List[int]]:
    # lane -> [v20..v28]
    vals: Dict[int, List[int]] = {lane: [0] * 9 for lane in range(64)}
    # v20..v28 live at word 0..8 in the 16-dword slot
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = int(row["tid"])
            lane = tid & 63
            word = int(row["word"])
            if word > 8:
                continue
            vals[lane][word] = int(row["val_u32"])
    return vals


def scaffold_read_addrs_for_lane(v20_28: List[int]) -> List[int]:
    v20, v21, v22, v23, v24, v25, v26, v27, v28 = v20_28
    addrs: List[int] = []
    # Matches scaffold ds_read_b64_tr_b8 sequence
    addrs += [v20 + o for o in (0, 256, 512, 768)]
    addrs += [v21 + 1024, v22 + 1152, v23 + 1280, v24 + 1408]
    addrs += [v20 + o for o in (2048, 2176, 2304, 2432)]
    addrs += [v25 + 3072, v26 + 3200, v27 + 3328, v28 + 3456]
    return addrs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=lambda x: int(x, 0), default=0x70)
    ap.add_argument("--ttbl", type=lambda x: int(x, 0), default=0x78)
    args = ap.parse_args()

    base_csv = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/scaffold_tr8_addrs.csv"
    bases = load_scaffold_bases(base_csv)

    # Build set of byte addresses written by preload stage (assume 16B stores)
    write_bytes: Set[int] = set()
    V_LDS0 = 41984
    for tid in range(256):
        b = V_LDS0 + write_addr_bitop3(tid, args.c, args.ttbl)
        # row r (16B)
        for i in range(16):
            write_bytes.add(b + i)
            write_bytes.add(b + 4096 + i)

    # Count read addresses (address is the base operand for ds_read_b64_tr_b8)
    # We consider it "covered" if that address points at a byte we wrote.
    total = 0
    covered = 0
    per_lane: List[Tuple[int, int]] = []
    for lane in range(64):
        rs = scaffold_read_addrs_for_lane(bases[lane])
        c = sum(1 for a in rs if a in write_bytes)
        per_lane.append((lane, c))
        total += len(rs)
        covered += c

    per_lane.sort(key=lambda x: x[1])
    print(f"total_reads={total} covered={covered} missing={total-covered}")
    print("worst 8 lanes:", per_lane[:8])
    print("best 8 lanes:", per_lane[-8:])


if __name__ == "__main__":
    main()


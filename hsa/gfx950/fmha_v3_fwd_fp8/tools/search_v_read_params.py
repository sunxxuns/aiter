#!/usr/bin/env python3
"""
Brute-force small knob search for TR8 V-read mapping quality using the existing
GPU oracle `test_v_read_map.py`.

We use V_READ_MAP=rowbyte + V_READ_RAW=1 so the output encodes the true global
row id as an 8-bit byte (per element), making address changes observable.

Goal: find simple tweaks (row-permutation choice via V_READ_CB, base XOR, etc.)
that make packed k->row mapping closer to identity:
  - lane 0:    row == k
  - lane 32:   row == 32 + k
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path("/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8")
ORACLE = BASE_DIR / "test_v_read_map.py"
PACKED_CSV = BASE_DIR / "v_read_packed.csv"


@dataclass(frozen=True)
class Candidate:
    score: int
    score32: int
    lane_xor: int
    base_xor: int
    s25_mask: int
    s25_override: int


def run_oracle(env: Dict[str, str]) -> None:
    full_env = os.environ.copy()
    full_env.update(env)
    subprocess.check_call(
        ["python3", str(ORACLE)],
        env=full_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_packed_csv() -> List[List[int]]:
    # rows[lane][k] = row_id
    rows: List[List[int]] = [[0] * 32 for _ in range(64)]
    with open(PACKED_CSV, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("lane") or not row.get("k") or not row.get("row"):
                continue
            try:
                lane = int(row["lane"])
                k = int(row["k"])
                rid = int(row["row"])
            except ValueError:
                continue
            if 0 <= lane < 64 and 0 <= k < 32:
                rows[lane][k] = rid
    return rows


def score_identity(rows: List[int], base: int = 0) -> int:
    # higher is better: count matches to identity (row == base + k)
    return sum(1 for k, r in enumerate(rows) if r == (base + k))


def main() -> None:
    base_xors = [0x0, 0x20, 0x60, 0x420, 0x460, 0x1020, 0x1060, 0x1420, 0x1460]
    perm_ids = [0, 1, 2, 3, 4, 5]  # encoded as (perm_id<<2) in V_READ_CB
    # encoded as (write_mode<<8) in V_READ_CB; mode2 selects solver-derived write swizzle (bitop3 0x7a,c=0)
    # Focused search: we care most about write_mode=2 (bitop3 ttbl=0x7a, c=0x0),
    # since it matches the scaffold TR8 reads much better than the baseline.
    write_modes = [2]
    # Keep a small but meaningful set of s25 overrides (these control which lane bits survive in tr8_base).
    s25_overrides = [0xB80, 0xFF0, 0xFFF]

    best: List[Candidate] = []
    total = 0
    # Focused search: vary s25 override as well (affects whether row LSB is preserved)
    for write_mode in write_modes:
        for perm_id in perm_ids:
            for s25_ov in s25_overrides:
                for base_xor in base_xors:
                    total += 1
                    env = {
                        "V_READ_MAP": "rowbyte",
                        "V_READ_RAW": "1",
                        "V_READ_SK": "128",
                        "V_READ_TILE": "0",
                        "V_READ_CB": str((perm_id << 2) | (write_mode << 8)),
                        "V_READ_BASE_XOR": hex(base_xor),
                        "V_READ_S25_OVERRIDE": hex(s25_ov),
                    }
                    run_oracle(env)
                    packed = load_packed_csv()
                    s0 = score_identity(packed[0], base=0)
                    s32 = score_identity(packed[32], base=32)
                    cand = Candidate(
                        score=s0, score32=s32,
                        lane_xor=write_mode,   # reuse to record write_mode
                        base_xor=base_xor,
                        s25_mask=s25_ov,       # reuse to record s25_override
                        s25_override=perm_id,  # reuse to record perm_id
                    )
                    best.append(cand)
                    best.sort(key=lambda c: (c.score + c.score32, c.score, c.score32), reverse=True)
                    best = best[:10]
                    print(
                        f"[{total:03d}] write_mode={write_mode} perm_id={perm_id} s25_ov={hex(s25_ov)} base_xor={hex(base_xor)} -> "
                        f"score0={s0}/32 score32={s32}/32 best_sum={best[0].score + best[0].score32}"
                    )

    print("\nTop candidates:")
    for i, c in enumerate(best):
        print(
            f"[{i}] sum={c.score + c.score32} score0={c.score} score32={c.score32} "
            f"write_mode={c.lane_xor} perm_id={c.s25_override} s25_ov={hex(c.s25_mask)} base_xor={hex(c.base_xor)}"
        )


if __name__ == "__main__":
    main()


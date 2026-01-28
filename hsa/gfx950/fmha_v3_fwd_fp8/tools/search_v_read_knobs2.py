#!/usr/bin/env python3
"""
Bounded, deterministic knob search for PV TR8 A-operand mapping.

Uses the GPU oracle `test_v_read_map.py` in a fast/quiet mode:
  - V_READ_MAP=rowbyte, V_READ_RAW=1 (true row id is byte value)
  - V_READ_QUIET=1, V_READ_SAVE_PACKED_ONLY=1

Objective:
  maximize identity matches in packed A regs:
    lane 0:  row == k
    lane 32: row == 32 + k
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path("/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8")
ORACLE = BASE_DIR / "test_v_read_map.py"
PACKED_CSV = BASE_DIR / "v_read_packed.csv"


@dataclass(frozen=True)
class Cand:
    score_sum: int
    score0: int
    score32: int
    perm_id: int
    v_read_offset: int
    v_read_v4_add: int
    v_read_lane_xor: int
    v_read_base_xor: int


def run_oracle(env: Dict[str, str]) -> None:
    full_env = os.environ.copy()
    full_env.update(env)
    subprocess.check_call(
        ["python3", str(ORACLE)],
        env=full_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_lane_rows(lane: int) -> List[int]:
    rows = [0] * 32
    with open(PACKED_CSV, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["lane"]) != lane:
                continue
            k = int(row["k"])
            rows[k] = int(row["row"])
    return rows


def score_identity(rows: List[int], base: int) -> int:
    return sum(1 for k, rid in enumerate(rows) if rid == base + k)


def main() -> None:
    # Search grid (small but meaningful)
    perm_ids = [0, 1]
    offsets = [0, 1, 2, 3]
    v4_adds = [0, 16, 32, 48]
    lane_xors = [0, 0x20]
    base_xors = [0, 0x20, 0x60, 0x420, 0x460]

    best: List[Cand] = []
    total = 0

    for perm_id in perm_ids:
        for off in offsets:
            for v4_add in v4_adds:
                for lane_xor in lane_xors:
                    for base_xor in base_xors:
                        total += 1
                        env = {
                            "V_READ_MAP": "rowbyte",
                            "V_READ_RAW": "1",
                            "V_READ_SK": "128",
                            "V_READ_TILE": "0",
                            "V_READ_QUIET": "1",
                            "V_READ_SAVE_PACKED_ONLY": "1",
                            # perm selector is in low bits of V_READ_CB (perm_id<<2)
                            "V_READ_CB": str(perm_id << 2),
                            "V_READ_OFFSET": str(off),
                            "V_READ_V4_ADD": str(v4_add),
                            "V_READ_LANE_XOR": hex(lane_xor),
                            "V_READ_BASE_XOR": hex(base_xor),
                        }
                        run_oracle(env)
                        lane0 = load_lane_rows(0)
                        lane32 = load_lane_rows(32)
                        s0 = score_identity(lane0, base=0)
                        s32 = score_identity(lane32, base=32)
                        cand = Cand(
                            score_sum=s0 + s32,
                            score0=s0,
                            score32=s32,
                            perm_id=perm_id,
                            v_read_offset=off,
                            v_read_v4_add=v4_add,
                            v_read_lane_xor=lane_xor,
                            v_read_base_xor=base_xor,
                        )
                        best.append(cand)
                        best.sort(key=lambda c: (c.score_sum, c.score0, c.score32), reverse=True)
                        best = best[:10]
                        if total % 20 == 0:
                            top = best[0]
                            print(
                                f"[{total:04d}] top_sum={top.score_sum} "
                                f"(perm={top.perm_id} off={top.v_read_offset} v4_add={top.v_read_v4_add} "
                                f"lane_xor={hex(top.v_read_lane_xor)} base_xor={hex(top.v_read_base_xor)})"
                            )

    print("\nTop candidates:")
    for i, c in enumerate(best):
        print(
            f"[{i}] sum={c.score_sum} (lane0={c.score0}/32 lane32={c.score32}/32) "
            f"perm={c.perm_id} off={c.v_read_offset} v4_add={c.v_read_v4_add} "
            f"lane_xor={hex(c.v_read_lane_xor)} base_xor={hex(c.v_read_base_xor)}"
        )


if __name__ == "__main__":
    main()


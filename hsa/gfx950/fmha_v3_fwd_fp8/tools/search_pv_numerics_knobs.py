#!/usr/bin/env python3
"""
Oracle-driven search over PV TR8 knobs by minimizing numerics error.

We run `test_scaffold_numerics.py` as the oracle (QK+PV full path) and parse max_err.
This is slow but robust: it directly optimizes correctness.
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys


NUMERICS = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/test_scaffold_numerics.py"


def run_one(env_overrides: dict[str, str]) -> float:
    env = os.environ.copy()
    env.update(env_overrides)
    p = subprocess.run(
        [sys.executable, NUMERICS],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    m = re.search(r"max_err:\s*([0-9eE\+\-\.]+|nan)", p.stdout, re.IGNORECASE)
    if not m:
        raise RuntimeError("failed to parse max_err from output:\\n" + p.stdout[-4000:])
    s = m.group(1).lower()
    val = float("nan") if s == "nan" else float(s)
    return val, p.stdout


def main():
    # Always use v_read_dump-style PV base path so v_read_cb knobs take effect.
    base_flags = 0x00000004

    # Small, targeted grid first (expand later)
    write_modes = [0, 2]  # 0x78 vs 0x7a
    perm_disable = [0, 0x00000080]  # keep both for now
    s25s = [0xB80, 0xFF0]
    v4_adds = [0, 16, 32]
    base_xors = [0, 0x20, 0x60, 0x420, 0x460, 0x1060, 0x1460]

    best = (1e9, None, "")
    tried = 0
    for wm, pd, s25, v4a, bx in itertools.product(write_modes, perm_disable, s25s, v4_adds, base_xors):
        tried += 1
        v_read_cb = (1 << 2) | (wm << 8)  # perm_id=1 (nontrivial), write_mode=wm
        flags = base_flags | pd
        env = {
            "NUMERICS_DEBUG_FLAGS": hex(flags),
            "NUMERICS_V_READ_CB": hex(v_read_cb),
            "NUMERICS_V_READ_S25_OVERRIDE": hex(s25),
            "NUMERICS_V_READ_V4_ADD": str(v4a),
            "NUMERICS_V_READ_BASE_XOR": hex(bx),
            "NUMERICS_V_READ_LANE_ADD": "0",
            "NUMERICS_V_READ_V3_XOR": "0",
            "NUMERICS_V_READ_V3_ADD": "0",
            "NUMERICS_V_READ_V2_ADD": "0",
            "NUMERICS_V_READ_BASE_ADD": "0",
            "NUMERICS_V_READ_BASE_EXTRA_ADD": "0",
        }
        val, out = run_one(env)
        score = 1e9 if (val != val) else val  # NaN -> huge
        if score < best[0]:
            best = (val, env, out)
            print(f"[best] max_err={val} env={env}")
        if tried % 10 == 0:
            print(f"[{tried}] current max_err={val}")

    print("\n=== BEST ===")
    print("max_err:", best[0])
    print("env:", best[1])
    print(best[2])


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Repro: scaffold PV TR8 dump (deterministic, one command).

This is a thin wrapper around:
  - build.sh
  - tools/dump_scaffold_tr8_raw_and_packed.py

Goal:
  Produce a single log file that captures:
  - the exact knobs used (env)
  - raw TR8 regs coverage
  - packed bytes checks

Typical use (copy/paste):
  python3 repro_scaffold_tr8.py

Knobs (env):
  HIP_VISIBLE_DEVICES (default "0")
  DUMP_COLBLK, DUMP_PERM_ID, DUMP_WRITE_MODE, DUMP_IDENTITY_WRITE, ...
  (forwarded to tools/dump_scaffold_tr8_raw_and_packed.py)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUILD = HERE / "build.sh"
DUMP = HERE / "tools" / "dump_scaffold_tr8_raw_and_packed.py"
OUT_DIR = HERE / "out" / "repro_scaffold_tr8"


def _env_snapshot(keys: list[str]) -> str:
    lines: list[str] = []
    for k in keys:
        if k in os.environ:
            lines.append(f"{k}={os.environ[k]}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"{run_id}.log"

    # Keep the run deterministic by default (matches the dump tool intent).
    os.environ.setdefault("DUMP_IDENTITY_WRITE", "1")
    os.environ.setdefault("DUMP_COLBLK", "0")
    os.environ.setdefault("DUMP_PERM_ID", "0")
    os.environ.setdefault("DUMP_WRITE_MODE", "2")
    os.environ.setdefault("DUMP_K_TILES", "2")

    forwarded_prefixes = (
        "HIP_",
        "DUMP_",
        "NUMERICS_",
    )
    forwarded_keys = sorted(
        [k for k in os.environ.keys() if k.startswith(forwarded_prefixes)]
    )

    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== repro_scaffold_tr8.py ===\n")
        f.write(f"cwd={HERE}\n")
        f.write(f"log={log_path}\n\n")
        f.write("=== env (forwarded) ===\n")
        f.write(_env_snapshot(forwarded_keys))
        f.write("\n")

        f.write("=== build ===\n")
        f.flush()
        subprocess.check_call(["bash", str(BUILD)], cwd=str(HERE), stdout=f, stderr=f)

        f.write("\n=== dump ===\n")
        f.flush()
        subprocess.check_call(
            [sys.executable, str(DUMP)],
            cwd=str(HERE),
            stdout=f,
            stderr=f,
            env=os.environ.copy(),
        )

    print(f"[ok] wrote {log_path}")


if __name__ == "__main__":
    main()


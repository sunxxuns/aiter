#!/usr/bin/env python3
"""
Dump TR8 read/write coverage maps to CSV.

Outputs:
  - tr8_reads.csv: lane,addr
  - v_writes.csv: lane,addr
  - tr8_coverage.csv: lane,total_reads,covered_reads,missing_reads
"""
from __future__ import annotations

import argparse
import csv
from typing import Iterable, Set

from tr8_layout_solver import (
    tr8_read_addrs,
    v_write_addrs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0xA400)
    parser.add_argument("--s25", type=lambda x: int(x, 0), default=0xB80)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--write-offsets", type=str, default="0,4096")
    parser.add_argument("--use-tr8-base", action="store_true")
    parser.add_argument("--base-xor", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    write_offsets = [int(x, 0) for x in args.write_offsets.split(",") if x.strip()]

    reads_path = f"{args.out_dir}/tr8_reads.csv"
    writes_path = f"{args.out_dir}/v_writes.csv"
    cov_path = f"{args.out_dir}/tr8_coverage.csv"

    all_writes: Set[int] = set()
    per_lane_reads = {}

    for lane in range(args.lanes):
        reads = tr8_read_addrs(lane, args.s25, args.base)
        per_lane_reads[lane] = reads
        for addr in v_write_addrs(
            lane,
            args.base,
            write_offsets,
            use_tr8_base=args.use_tr8_base,
            s25=args.s25,
            base_xor=args.base_xor,
        ):
            all_writes.add(addr)

    with open(reads_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["lane", "addr"])
        for lane, reads in per_lane_reads.items():
            for addr in sorted(reads):
                w.writerow([lane, addr])

    with open(writes_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["lane", "addr"])
        for lane in range(args.lanes):
            for addr in sorted(
                v_write_addrs(
                    lane,
                    args.base,
                    write_offsets,
                    use_tr8_base=args.use_tr8_base,
                    s25=args.s25,
                    base_xor=args.base_xor,
                )
            ):
                w.writerow([lane, addr])

    with open(cov_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["lane", "total_reads", "covered_reads", "missing_reads"])
        for lane, reads in per_lane_reads.items():
            covered = sum(1 for a in reads if a in all_writes)
            w.writerow([lane, len(reads), covered, len(reads) - covered])

    print(f"wrote {reads_path}")
    print(f"wrote {writes_path}")
    print(f"wrote {cov_path}")


if __name__ == "__main__":
    main()

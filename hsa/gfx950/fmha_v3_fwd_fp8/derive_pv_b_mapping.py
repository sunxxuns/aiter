#!/usr/bin/env python3
"""Derive PV B-operand mapping diagnostics from p_pack_mapping.csv.

This script summarizes how packed P bytes (post-mix) map to (row, k),
and emits a markdown report for debugging the PV B-operand layout.
"""
from __future__ import annotations

import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path("/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8")
CSV_PATH = BASE_DIR / "p_pack_mapping.csv"
REPORT_PATH = BASE_DIR / "pv_b_mapping_report.md"


def load_mapping():
    rows = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "row": int(row["row"]),
                    "k": int(row["k"]),
                    "src_lane": int(row["src_lane"]),
                    "src_reg": int(row["src_reg"]),
                    "src_byte": int(row["src_byte"]),
                }
            )
    return rows


def summarize(rows):
    valid = [r for r in rows if r["src_lane"] >= 0]
    missing = len(rows) - len(valid)

    # Row LSB vs byte parity correlation.
    match_parity = sum(
        1 for r in valid if (r["src_byte"] & 1) == (r["row"] & 1)
    )

    # Diagonal (row == k) mapping.
    diag = [
        r for r in valid if r["row"] == r["k"] and r["row"] < 32 and r["k"] < 32
    ]
    diag_map = {
        r["row"]: (r["src_lane"], r["src_reg"] * 4 + r["src_byte"])
        for r in diag
    }

    # Lane distribution for diag entries.
    diag_lane_counts = Counter(lane for lane, _ in diag_map.values())

    # Per-row src_byte distribution.
    row_byte_dist = defaultdict(Counter)
    for r in valid:
        row_byte_dist[r["row"]][r["src_byte"]] += 1

    return {
        "missing": missing,
        "valid": len(valid),
        "match_parity": match_parity,
        "diag_map": diag_map,
        "diag_lane_counts": diag_lane_counts,
        "row_byte_dist": row_byte_dist,
    }


def write_report(summary):
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    lines.append("# PV B-Operand Mapping Report")
    lines.append("")
    lines.append(f"- Generated: `{ts}`")
    lines.append(f"- Source CSV: `{CSV_PATH}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Valid mappings: `{summary['valid']}`")
    lines.append(f"- Missing mappings: `{summary['missing']}`")
    lines.append("")
    lines.append("## Row LSB vs Byte Parity")
    lines.append(
        f"- `(src_byte & 1) == (row & 1)` matches: "
        f"`{summary['match_parity']}/{summary['valid']}`"
    )
    lines.append(
        "- Interpretation: row LSB is encoded in byte parity after mix."
    )
    lines.append("")
    lines.append("## Diagonal Mapping (row == k, 0..31)")
    lines.append("Format: `row -> (lane, byte_pos)` where byte_pos = src_reg*4 + src_byte.")
    lines.append("")
    for row in range(32):
        if row in summary["diag_map"]:
            lane, pos = summary["diag_map"][row]
            lines.append(f"- `{row:02d} -> (lane {lane:02d}, pos {pos:02d})`")
        else:
            lines.append(f"- `{row:02d} -> (missing)`")
    lines.append("")
    lines.append("## Diagonal Lane Distribution")
    for lane, count in sorted(summary["diag_lane_counts"].items()):
        lines.append(f"- lane {lane:02d}: {count}")
    lines.append("")
    lines.append("## Per-Row Byte Histogram (rows 0..7)")
    for row in range(8):
        dist = summary["row_byte_dist"].get(row, {})
        if not dist:
            lines.append(f"- row {row:02d}: (missing)")
            continue
        buckets = ", ".join(f"b{b}:{dist[b]}" for b in sorted(dist))
        lines.append(f"- row {row:02d}: {buckets}")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Row LSB parity is preserved in byte parity, but identity-P rowid still "
        "collapses, implying MFMA B expects a different byte placement."
    )
    lines.append(
        "- Diagonal entries show lane remapping for row 0 and rows 16..31; "
        "this likely indicates missing lane/byte permute after mix."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing mapping CSV: {CSV_PATH}")
    rows = load_mapping()
    summary = summarize(rows)
    write_report(summary)
    print(f"Wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

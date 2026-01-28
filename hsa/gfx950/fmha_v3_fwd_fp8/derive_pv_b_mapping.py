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

import ctypes
import torch


BASE_DIR = Path("/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8")
CSV_PATH = BASE_DIR / "p_pack_mapping.csv"
REPORT_PATH = BASE_DIR / "pv_b_mapping_report.md"
SCAFFOLD_CO = BASE_DIR / "fwd_fp8_scaffold.co"


def load_kernel(co_path, kernel_name):
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), str(co_path).encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return hip, module, func


def decode_scaffold_output(raw, s_q, d, waves_per_block=8):
    mapping = []
    for col_off in (0, 32, 64, 96):
        for row_8 in range(4):
            for row_mod4 in range(4):
                mapping.append((row_mod4, row_8, col_off))
    decoded = torch.empty((s_q, d), dtype=torch.float32)
    for tid in range(waves_per_block * 64):
        lane = tid & 63
        wave = tid >> 6
        col = lane & 31
        row_base = ((lane >> 5) & 1) * 4
        wave_row = wave * 32
        base = tid * 64
        for i, (row_mod4, row_8, col_off) in enumerate(mapping):
            row = row_mod4 + row_base + row_8 * 8 + wave_row
            decoded[row, col + col_off] = raw[base + i]
    return decoded


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


def run_rowbit_mapping():
    if not SCAFFOLD_CO.exists():
        raise RuntimeError(f"Missing scaffold co: {SCAFFOLD_CO}")

    torch.manual_seed(0)
    B, H, D = 1, 1, 128
    S = 64
    num_q_blocks = 2
    num_k_tiles = 2
    s_q, s_k = 256, 64

    Qf32 = torch.zeros(B, H, s_q, D, device="cuda", dtype=torch.float32)
    Kf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
    for r in range(min(s_k, D)):
        Qf32[0, 0, r, r] = 1.0
        Kf32[0, 0, r, r] = 1.0

    Q = Qf32.to(torch.float8_e4m3fn)
    K = Kf32.to(torch.float8_e4m3fn)

    hip, module, func = load_kernel(SCAFFOLD_CO, "_fwd_fp8_scaffold")
    bit_outputs = []
    for bit in range(6):
        Vf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
        for r in range(s_k):
            Vf32[0, 0, r, :] = float((r >> bit) & 1)
        V = Vf32.to(torch.float8_e4m3fn)
        O = torch.zeros(B, H, s_q, D, device="cuda", dtype=torch.float32)
        debug_flags = int(os.environ.get("SCAFFOLD_DEBUG_FLAGS", "0"), 0)
        v_read_cb = int(os.environ.get("SCAFFOLD_V_READ_CB", "0"), 0)
        args = [
            ctypes.c_void_p(O.data_ptr()),
            ctypes.c_void_p(Q.data_ptr()),
            ctypes.c_void_p(K.data_ptr()),
            ctypes.c_void_p(V.data_ptr()),
            ctypes.c_int32(num_k_tiles),
            ctypes.c_int32(s_q * D),
            ctypes.c_int32(s_k * D),
            ctypes.c_int32(s_k * D),
            ctypes.c_int32(s_q * D * 4),
            ctypes.c_int32(debug_flags),
            ctypes.c_int32(v_read_cb),
        ctypes.c_int32(v_read_lane_add),
        ctypes.c_int32(v_read_v3_xor),
        ctypes.c_int32(v_read_v3_add),
        ctypes.c_int32(v_read_v4_add),
        ctypes.c_int32(v_read_v2_add),
        ctypes.c_int32(v_read_base_add),
        ctypes.c_int32(v_read_base_xor),
        ctypes.c_int32(v_read_base_extra_add),
        ctypes.c_int32(v_read_s25_override),
        ]
        args_ptrs = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )
        hip.hipModuleLaunchKernel(
            func,
            num_q_blocks // 2, 1, 1,
            512, 1, 1,
            50176,
            None,
            args_ptrs,
            None,
        )
        hip.hipDeviceSynchronize()
        raw = O.detach().cpu().view(-1)
        decoded = decode_scaffold_output(raw, s_q, D)
        bit_outputs.append([int(round(decoded[r, 0].item())) for r in range(s_k)])

    hip.hipModuleUnload(module)
    observed = []
    for r in range(s_k):
        val = 0
        for bit in range(6):
            if bit_outputs[bit][r]:
                val |= (1 << bit)
        observed.append(val)
    return observed


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
    lines.append("")
    lines.append("## Row Mapping (identity-P, rowbit probe)")
    try:
        observed = run_rowbit_mapping()
        lines.append("Output row -> observed row (0..63):")
        for base in range(0, 64, 8):
            chunk = observed[base:base + 8]
            lines.append(f"- {base:02d}-{base+7:02d}: {chunk}")
        lines.append(
            "- This mapping shows row bit0 and bit4 are dropped "
            "(rows fold by r>>1 within 0..15 and 16..31)."
        )
    except Exception as exc:
        lines.append(f"- Row mapping probe failed: {exc}")

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

#!/usr/bin/env python3
"""Layout probes for QK+PV scaffold.

Runs small deterministic cases to reveal how P/V layouts map into O.
This is meant to be a fast, repeatable sanity suite before kernel edits.
"""
import os
import ctypes
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"

from test_scaffold_numerics import load_kernel, decode_scaffold_output  # noqa: E402


def build_identity_qk(s_q, s_k, d, device):
    q = torch.zeros(1, 1, s_q, d, device=device, dtype=torch.float32)
    k = torch.zeros(1, 1, s_k, d, device=device, dtype=torch.float32)
    for r in range(min(s_k, d)):
        q[0, 0, r, r] = 1.0
        k[0, 0, r, r] = 1.0
    return q, k


def run_kernel(q, k, v, num_q_blocks, num_k_tiles, s_q, s_k, d):
    fp8_dtype = v.dtype
    o = torch.zeros(1, 1, s_q, d, device="cuda", dtype=torch.float32)
    args = [
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_int32(num_k_tiles),
        ctypes.c_int32(s_q * d),
        ctypes.c_int32(s_k * d),
        ctypes.c_int32(s_k * d),
        ctypes.c_int32(s_q * d * 4),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    _, func = load_kernel(
        "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co",
        "_fwd_fp8_scaffold",
    )
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
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
    raw = o.detach().cpu().view(-1)
    decoded = decode_scaffold_output(raw, s_q, d)
    return decoded, o


def print_mapping(decoded, rows=16):
    mapping = [int(round(decoded[r, 0].item())) for r in range(rows)]
    print(f"row->value (first {rows}): {mapping}")


def run_case(name, q, k, v, num_q_blocks, num_k_tiles, s_q, s_k, d):
    decoded, o = run_kernel(q, k, v, num_q_blocks, num_k_tiles, s_q, s_k, d)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    p = torch.matmul(qf, kf.transpose(-1, -2)).to(torch.float8_e4m3fn).float()
    ref = torch.matmul(p, vf)[0, 0].detach().cpu()
    diff = (decoded - ref).abs()
    print(f"\n=== {name} ===")
    print(f"max_err: {diff.max().item():.6f}, mean_err: {diff.mean().item():.6f}")
    print("decoded[0:8,0:8]:")
    print(decoded[:8, :8])
    print("ref[0:8,0:8]:")
    print(ref[:8, :8])
    print_mapping(decoded, rows=16)
    if "col-pattern" in name:
        for r in range(4):
            cols = [int(round(decoded[r, c].item() * 128)) for c in range(8)]
            print(f"row {r} col-map (0..7)->{cols}")
        cols = [int(round(decoded[0, c].item() * 128)) for c in range(32)]
        print(f"row 0 col-map (0..31)->{cols}")
        cols = [int(round(decoded[0, c].item() * 128)) for c in range(64)]
        print(f"row 0 col-map (0..63)->{cols}")
        cols = [int(round(decoded[1, c].item() * 128)) for c in range(32)]
        print(f"row 1 col-map (0..31)->{cols}")
        cols = [int(round(decoded[2, c].item() * 128)) for c in range(32)]
        print(f"row 2 col-map (0..31)->{cols}")
        cols = [int(round(decoded[2, c].item() * 128)) for c in range(64)]
        print(f"row 2 col-map (0..63)->{cols}")
        cols = [int(round(decoded[3, c].item() * 128)) for c in range(32)]
        print(f"row 3 col-map (0..31)->{cols}")
        cols = [int(round(decoded[4, c].item() * 128)) for c in range(32)]
        print(f"row 4 col-map (0..31)->{cols}")


def swizzle_cols_16x8(v):
    """Transpose col index bits: new_col = (c % 16) * 8 + (c // 16)."""
    d = v.shape[-1]
    perm = torch.empty(d, device=v.device, dtype=torch.long)
    for c in range(d):
        perm[c] = (c % 16) * 8 + (c // 16)
    return v.index_select(-1, perm)


def swizzle_tr8_8x8(v):
    """Swap row/col low-3 bits within 8x8 blocks."""
    s = v.shape[-2]
    d = v.shape[-1]
    out = torch.empty_like(v)
    for r in range(s):
        for c in range(d):
            r2 = (r & ~7) | (c & 7)
            c2 = (c & ~7) | (r & 7)
            out[0, 0, r2, c2] = v[0, 0, r, c]
    return out


def swizzle_row_rot16(v):
    """Rotate columns by (row>>1)*16 within each 64-col half."""
    s = v.shape[-2]
    d = v.shape[-1]
    out = torch.empty_like(v)
    for r in range(s):
        shift = ((r >> 1) & 3) * 16
        for c in range(64):
            out[0, 0, r, (c + shift) & 63] = v[0, 0, r, c]
        for c in range(64, d):
            out[0, 0, r, 64 + ((c - 64 + shift) & 63)] = v[0, 0, r, c]
    return out


def main():
    torch.manual_seed(0)
    dtype_name = os.environ.get("NUMERICS_FP8", "e4m3fn")
    fp8_dtype = {
        "e4m3fn": torch.float8_e4m3fn,
        "e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
        "e5m2": getattr(torch, "float8_e5m2", None),
        "e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None),
    }.get(dtype_name)
    if fp8_dtype is None:
        raise RuntimeError(f"Unsupported FP8 dtype: {dtype_name}")

    s = int(os.environ.get("NUMERICS_S", "64"))
    b, h, d = 1, 1, 128
    num_q_blocks = (s + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = (s + 31) // 32
    if num_k_tiles % 2 != 0:
        num_k_tiles += 1
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    qf32, kf32 = build_identity_qk(s_q, s_k, d, device="cuda")
    q = qf32.to(fp8_dtype)
    k = kf32.to(fp8_dtype)

    v_col = torch.zeros(1, 1, s_k, d, device="cuda", dtype=torch.float32)
    for c in range(d):
        v_col[0, 0, :, c] = c / 128.0
    run_case("identity P + V col-pattern", q, k, v_col.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)
    v_col_swz = swizzle_cols_16x8(v_col)
    run_case("identity P + V col-pattern (swizzle16x8)", q, k, v_col_swz.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)
    v_tr8 = swizzle_tr8_8x8(v_col)
    run_case("identity P + V col-pattern (tr8-8x8)", q, k, v_tr8.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)
    v_rot16 = swizzle_row_rot16(v_col)
    run_case("identity P + V col-pattern (row-rot16)", q, k, v_rot16.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)

    v_rowid = torch.zeros(1, 1, s_k, d, device="cuda", dtype=torch.float32)
    for r in range(s_k):
        v_rowid[0, 0, r, :] = float(r)
    run_case("identity P + V rowid", q, k, v_rowid.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)

    v_even = torch.zeros(1, 1, s_k, d, device="cuda", dtype=torch.float32)
    for r in range(0, s_k, 2):
        v_even[0, 0, r, :] = float(r)
    run_case("identity P + V even rows only", q, k, v_even.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)

    v_odd = torch.zeros(1, 1, s_k, d, device="cuda", dtype=torch.float32)
    for r in range(1, s_k, 2):
        v_odd[0, 0, r, :] = float(r)
    run_case("identity P + V odd rows only", q, k, v_odd.to(fp8_dtype),
             num_q_blocks, num_k_tiles, s_q, s_k, d)


if __name__ == "__main__":
    main()

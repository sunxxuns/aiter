#!/usr/bin/env python3
"""Recover column mapping via bit-encoded V patterns."""
import ctypes
import torch

from test_scaffold_numerics import load_kernel, decode_scaffold_output  # noqa: E402


def build_identity_qk(s_q, s_k, d, device):
    q = torch.zeros(1, 1, s_q, d, device=device, dtype=torch.float32)
    k = torch.zeros(1, 1, s_k, d, device=device, dtype=torch.float32)
    for r in range(min(s_k, d)):
        q[0, 0, r, r] = 1.0
        k[0, 0, r, r] = 1.0
    return q, k


def run_kernel(q, k, v, num_q_blocks, num_k_tiles, s_q, s_k, d):
    o = torch.zeros(1, 1, s_q, d, device="cuda", dtype=torch.float32)
    debug_flags = int(os.environ.get("SCAFFOLD_DEBUG_FLAGS", "0"), 0)
    v_read_cb = int(os.environ.get("SCAFFOLD_V_READ_CB", "0"), 0)
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
    return decoded


def main():
    torch.manual_seed(0)
    s = 64
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
    q = qf32.to(torch.float8_e4m3fn)
    k = kf32.to(torch.float8_e4m3fn)

    bits = []
    for bit in range(7):
        v = torch.zeros(1, 1, s_k, d, device="cuda", dtype=torch.float32)
        for c in range(d):
            v[0, 0, :, c] = float((c >> bit) & 1)
        decoded = run_kernel(q, k, v.to(torch.float8_e4m3fn),
                             num_q_blocks, num_k_tiles, s_q, s_k, d)
        bits.append((decoded[:s, :d] > 0.5).to(torch.int32))

    col_map = torch.zeros((s, d), dtype=torch.int32)
    for bit, mat in enumerate(bits):
        col_map += (mat << bit)

    print("=== Column map summary ===")
    for r in range(8):
        row = col_map[r].tolist()
        uniq = len(set(row))
        print(f"row {r} unique cols: {uniq}, first 32: {row[:32]}")

    # Print a compact map for row 0 and row 2
    print("\nrow 0 map (0..63):", col_map[0][:64].tolist())
    print("row 2 map (0..63):", col_map[2][:64].tolist())


if __name__ == "__main__":
    main()

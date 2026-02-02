#!/usr/bin/env python3
"""Numerics check for QK+PV scaffold (no softmax).

This decodes the MFMA output layout into row-major O and compares
against a PyTorch reference that matches the kernel math:
O = (Q @ K^T) quantized to FP8, then O = P_fp8 @ V.
"""
import os
# Default to GPU2 unless the user already set it.
os.environ.setdefault("HIP_VISIBLE_DEVICES", "2")

import ctypes
import math
import torch

hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")


def load_kernel(co_path, kernel_name):
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return module, func


def decode_scaffold_output(raw, s_q, d, waves_per_block=8):
    """Decode thread-major MFMA outputs to row-major O (S_q x D)."""
    use_f = os.environ.get("NUMERICS_DECODE_F", "0") == "1"
    mapping = []
    for col_off in (0, 32, 64, 96):
        for row_8 in range(4):
            for row_mod4 in range(4):
                mapping.append((row_mod4, row_8, col_off))

    expected_len = waves_per_block * 64 * 64
    if raw.numel() != expected_len:
        raise ValueError(f"raw size {raw.numel()} != expected {expected_len}")

    decoded = torch.empty((s_q, d), dtype=torch.float32)
    for tid in range(waves_per_block * 64):
        lane = tid & 63
        wave = tid >> 6
        col = ((lane | (lane >> 1)) & 31) if use_f else (lane & 31)
        row_base = ((lane >> 5) & 1) * 4
        wave_row = wave * 32
        base = tid * 64
        for i, (row_mod4, row_8, col_off) in enumerate(mapping):
            row = row_mod4 + row_base + row_8 * 8 + wave_row
            decoded[row, col + col_off] = raw[base + i]
    return decoded


def main():
    # Small but non-trivial shape (one block, K tiles even)
    B, H, D = 1, 1, 128
    S = int(os.environ.get("NUMERICS_S", "64"))
    num_q_blocks = (S + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    env_k_tiles = os.environ.get("NUMERICS_K_TILES")
    if env_k_tiles is not None:
        num_k_tiles = int(env_k_tiles)
    else:
        num_k_tiles = (S + 31) // 32
        if num_k_tiles % 2 != 0:
            num_k_tiles += 1
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    print("=== QK+PV Scaffold Numerics Check ===")
    print(f"B={B}, H={H}, S={S}, D={D}")
    print(f"Q blocks={num_q_blocks}, K tiles={num_k_tiles}")
    print(f"S_q={s_q}, S_k={s_k}")

    torch.manual_seed(0)
    dtype_name = os.environ.get("NUMERICS_FP8", "e4m3fn")
    dtype_map = {
        "e4m3fn": torch.float8_e4m3fn,
        "e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
        "e5m2": getattr(torch, "float8_e5m2", None),
        "e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None),
    }
    fp8_dtype = dtype_map.get(dtype_name)
    if fp8_dtype is None:
        raise RuntimeError(f"Unsupported FP8 dtype: {dtype_name}")

    if os.environ.get("NUMERICS_IDENTITY_P", "0") == "1":
        Qf32 = torch.zeros(B, H, s_q, D, device="cuda", dtype=torch.float32)
        Kf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
        for r in range(min(s_k, D)):
            Qf32[0, 0, r, r] = 1.0
            Kf32[0, 0, r, r] = 1.0
        v_pattern = os.environ.get("NUMERICS_V_PATTERN", "row")
        Vf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
        if v_pattern == "col":
            for c in range(D):
                Vf32[0, 0, :, c] = c / 128.0
        elif v_pattern == "rowid":
            for r in range(s_k):
                Vf32[0, 0, r, :] = float(r)
        elif v_pattern == "rowbits":
            for r in range(s_k):
                Vf32[0, 0, r, :] = float(r & 0xF)
        elif v_pattern.startswith("rowbit"):
            try:
                bit = int(v_pattern.replace("rowbit", ""))
            except ValueError as exc:
                raise RuntimeError(f"Invalid rowbit pattern: {v_pattern}") from exc
            for r in range(s_k):
                Vf32[0, 0, r, :] = float((r >> bit) & 1)
        elif v_pattern == "rowcol":
            cols = (torch.arange(D, device="cuda", dtype=torch.float32) / 128.0)
            for r in range(s_k):
                Vf32[0, 0, r, :] = float(r) + cols
        else:
            for r in range(s_k):
                Vf32[0, 0, r, :] = r / 64.0
        Q = Qf32.to(fp8_dtype)
        K = Kf32.to(fp8_dtype)
        v_range = os.environ.get("NUMERICS_V_RANGE")
        if v_range:
            start_s, end_s = v_range.split(":")
            start_k = int(start_s)
            end_k = int(end_s)
            Vf32[:, :, :start_k, :] = 0
            Vf32[:, :, end_k:, :] = 0
        V = Vf32.to(fp8_dtype)
    elif os.environ.get("NUMERICS_ONES", "0") == "1":
        Q = torch.ones(B, H, s_q, D, device="cuda", dtype=torch.float32).to(fp8_dtype)
        K = torch.ones(B, H, s_k, D, device="cuda", dtype=torch.float32).to(fp8_dtype)
        V = torch.ones(B, H, s_k, D, device="cuda", dtype=torch.float32).to(fp8_dtype)
    else:
        Q = (torch.randn(B, H, s_q, D, device="cuda") * 0.1).to(fp8_dtype)
        K = (torch.randn(B, H, s_k, D, device="cuda") * 0.1).to(fp8_dtype)
        V = (torch.randn(B, H, s_k, D, device="cuda") * 0.1).to(fp8_dtype)

    # ---------------------------------------------------------------------
    # Operand-isolation overrides (apply after initial init path)
    # ---------------------------------------------------------------------
    if os.environ.get("NUMERICS_Q_ONES", "0") == "1":
        Q = torch.ones(B, H, s_q, D, device="cuda", dtype=torch.float32).to(fp8_dtype)
    if os.environ.get("NUMERICS_K_ONES", "0") == "1":
        K = torch.ones(B, H, s_k, D, device="cuda", dtype=torch.float32).to(fp8_dtype)
    if os.environ.get("NUMERICS_V_ONES", "0") == "1":
        V = torch.ones(B, H, s_k, D, device="cuda", dtype=torch.float32).to(fp8_dtype)
    if os.environ.get("NUMERICS_V_ROWID", "0") == "1":
        Vf32 = torch.empty(B, H, s_k, D, device="cuda", dtype=torch.float32)
        for r in range(s_k):
            Vf32[0, 0, r, :] = float(r)
        V = Vf32.to(fp8_dtype)
    if os.environ.get("NUMERICS_ZERO_Q", "0") == "1":
        Q.zero_()
    if os.environ.get("NUMERICS_ZERO_K", "0") == "1":
        K.zero_()
    if os.environ.get("NUMERICS_ZERO_V", "0") == "1":
        V.zero_()
    print(f"Q finite: {torch.isfinite(Q.float()).all().item()}")
    print(f"K finite: {torch.isfinite(K.float()).all().item()}")
    print(f"V finite: {torch.isfinite(V.float()).all().item()}")
    O = torch.zeros(B, H, s_q, D, device="cuda", dtype=torch.float32)

    stride_qh = s_q * D        # bytes for FP8
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = s_q * D * 4    # bytes for FP32
    debug_flags = int(os.environ.get("NUMERICS_DEBUG_FLAGS", "0"), 0)
    v_read_cb = int(os.environ.get("NUMERICS_V_READ_CB", "0"), 0)
    v_read_lane_add = int(os.environ.get("NUMERICS_V_READ_LANE_ADD", "0"), 0)
    v_read_v3_xor = int(os.environ.get("NUMERICS_V_READ_V3_XOR", "0"), 0)
    v_read_v3_add = int(os.environ.get("NUMERICS_V_READ_V3_ADD", "0"), 0)
    v_read_v4_add = int(os.environ.get("NUMERICS_V_READ_V4_ADD", "0"), 0)
    v_read_v2_add = int(os.environ.get("NUMERICS_V_READ_V2_ADD", "0"), 0)
    v_read_base_add = int(os.environ.get("NUMERICS_V_READ_BASE_ADD", "0"), 0)
    v_read_base_xor = int(os.environ.get("NUMERICS_V_READ_BASE_XOR", "0"), 0)
    v_read_base_extra_add = int(os.environ.get("NUMERICS_V_READ_BASE_EXTRA_ADD", "0"), 0)
    v_read_s25_override = int(os.environ.get("NUMERICS_V_READ_S25_OVERRIDE", "0"), 0)

    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_int32(num_k_tiles),
        ctypes.c_int32(stride_qh),
        ctypes.c_int32(stride_kh),
        ctypes.c_int32(stride_vh),
        ctypes.c_int32(stride_oh),
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

    co_path = os.environ.get(
        "SCAFFOLD_CO_PATH",
        "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co",
    )
    kernel_name = os.environ.get("SCAFFOLD_KERNEL_NAME", "_fwd_fp8_scaffold")
    _, func = load_kernel(co_path, kernel_name)

    grid = (num_q_blocks // 2, B * H, 1)
    block = (512, 1, 1)
    # IMPORTANT: `fwd_fp8_scaffold.s` declares `.amdhsa_group_segment_fixed_size 50176`.
    # Passing 50176 again as dynamic LDS would double-allocate LDS per workgroup.
    lds_bytes = 0

    hip.hipModuleLaunchKernel(
        func,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        lds_bytes,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()

    # Reference: P = Q*K^T, quantize to FP8, then O = P_fp8*V
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    Vf_ref = Vf
    if os.environ.get("NUMERICS_ZERO_ODD_V", "0") == "1":
        Vf_ref = Vf.clone()
        Vf_ref[:, :, 1::2, :] = 0
    if os.environ.get("NUMERICS_P_ONES", "0") == "1":
        P = torch.ones((B, H, s_q, s_k), device="cuda", dtype=torch.float32)
    else:
        P = torch.matmul(Qf, Kf.transpose(-1, -2))
    P_fp8 = P.to(torch.float8_e4m3fn)
    O_ref = torch.matmul(P_fp8.float(), Vf_ref)

    o_finite = torch.isfinite(O)
    nan_count = torch.isnan(O).sum().item()
    inf_count = torch.isinf(O).sum().item()
    print(f"kernel output finite: {o_finite.all().item()} (nan={nan_count}, inf={inf_count})")

    # Decode kernel output to row-major
    raw = O.detach().cpu().view(-1)
    raw_mat = raw.view(-1, 64)
    nan_per_thread = torch.isnan(raw_mat).sum(dim=1)
    threads_all_nan = (nan_per_thread == 64).sum().item()
    threads_no_nan = (nan_per_thread == 0).sum().item()
    decoded = decode_scaffold_output(raw, s_q, D)
    O_ref_cpu = O_ref[0, 0].detach().cpu()

    diff = (decoded - O_ref_cpu).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (O_ref_cpu.abs() + 1e-6)).max().item()
    finite_rows = torch.isfinite(decoded).all(dim=1).sum().item()
    if os.environ.get("NUMERICS_SHOW_ARGMAX", "0") == "1":
        flat_idx = int(diff.view(-1).argmax().item())
        r = flat_idx // D
        c = flat_idx % D
        print(f"argmax_err at (r={r}, c={c}): kernel={decoded[r, c].item():.6f} ref={O_ref_cpu[r, c].item():.6f} abs_err={diff[r, c].item():.6f}")

    print(f"ref output finite: {torch.isfinite(O_ref).all().item()}")
    print(f"max_err: {max_err:.6f}")
    print(f"mean_err: {mean_err:.6f}")
    print(f"max_rel_err: {rel_err:.6f}")
    print(f"decoded finite: {torch.isfinite(decoded).all().item()}")
    if os.environ.get("NUMERICS_IDENTITY_P", "0") == "1":
        print("decoded[0:8,0:8]:")
        print(decoded[:8, :8])
        print("ref[0:8,0:8]:")
        print(O_ref_cpu[:8, :8])
        if os.environ.get("NUMERICS_DUMP_ROWID_SLICE", "0") == "1":
            # Useful for spotting row-permutation/row-mapping issues in the decode.
            rs = list(range(24, 40))
            print("decoded[row,0] for rows 24..39:")
            print([float(decoded[r, 0].item()) for r in rs])
            print("ref[row,0] for rows 24..39:")
            print([float(O_ref_cpu[r, 0].item()) for r in rs])
    print(f"finite rows: {finite_rows}/{decoded.shape[0]}")
    print(f"threads all-NaN: {threads_all_nan}, no-NaN: {threads_no_nan}")
    waves_per_block = 8
    for wave in range(waves_per_block):
        wave_nan = nan_per_thread[wave * 64:(wave + 1) * 64]
        wave_all_nan = (wave_nan == 64).sum().item()
        wave_any_nan = (wave_nan > 0).sum().item()
        print(f"wave {wave}: threads with any NaN={wave_any_nan}, all NaN={wave_all_nan}")


if __name__ == "__main__":
    main()

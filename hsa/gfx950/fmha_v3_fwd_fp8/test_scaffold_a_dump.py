#!/usr/bin/env python3
"""Dump A regs (v48..v55) after P_A_TRANSPOSE overrides."""
import ctypes
import os
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"
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


def main():
    B, H, D = 1, 1, 128
    S = int(os.environ.get("NUMERICS_S", "64"))
    num_q_blocks = (S + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = int(os.environ.get("NUMERICS_K_TILES", "1"))
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    Qf32 = torch.zeros(B, H, s_q, D, device="cuda", dtype=torch.float32)
    Kf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
    for r in range(min(s_k, D)):
        Qf32[0, 0, r, r] = 1.0
        Kf32[0, 0, r, r] = 1.0

    V_pattern = os.environ.get("NUMERICS_V_PATTERN", "rowid")
    use_v_raw = os.environ.get("NUMERICS_V_RAW", "0") == "1"
    Vf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
    V_raw = None
    if V_pattern == "codebook32":
        table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        finite = table[torch.isfinite(table)]
        finite = finite[(finite > 0) & (finite <= 16.0)]
        unique = torch.unique(finite)
        unique, _ = torch.sort(unique)
        idx = torch.linspace(0, unique.numel() - 1, steps=32).round().long()
        codes = unique[idx]
        for r in range(s_k):
            Vf32[0, 0, r, :] = float(codes[r % 32].item())
    elif V_pattern == "colbyte":
        byte_lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device="cuda")
        col_bytes = torch.arange(D, dtype=torch.long, device="cuda")
        col_vals = byte_lut[col_bytes].float()
        Vf32[0, 0, :, :] = col_vals
    elif V_pattern == "blockk":
        col_bytes = torch.arange(D, dtype=torch.long, device="cuda")
        col_blocks = (col_bytes // 32) & 3
        if use_v_raw:
            v_bytes = torch.empty((s_k, D), dtype=torch.uint8, device="cuda")
            for r in range(s_k):
                k_id = r & 31
                byte_ids = ((col_blocks << 5) | k_id).to(torch.uint8)
                v_bytes[r, :] = byte_ids
            V_raw = v_bytes.view(torch.float8_e4m3fn)
            Vf32[0, 0, :, :] = V_raw.float()
        else:
            byte_lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device="cuda")
            for r in range(s_k):
                k_id = r & 31
                byte_ids = ((col_blocks << 5) | k_id).to(torch.long)
                Vf32[0, 0, r, :] = byte_lut[byte_ids].float()
    elif V_pattern == "kbyte":
        if use_v_raw:
            v_bytes = torch.empty((s_k, D), dtype=torch.uint8, device="cuda")
            for r in range(s_k):
                k_id = r & 31
                v_bytes[r, :] = k_id
            V_raw = v_bytes.view(torch.float8_e4m3fn)
            Vf32[0, 0, :, :] = V_raw.float()
        else:
            byte_lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device="cuda")
            for r in range(s_k):
                k_id = r & 31
                Vf32[0, 0, r, :] = byte_lut[k_id].float()
    elif V_pattern == "rowid":
        for r in range(s_k):
            Vf32[0, 0, r, :] = float(r)
    else:
        for r in range(s_k):
            Vf32[0, 0, r, :] = float(r)

    Q = Qf32.to(torch.float8_e4m3fn)
    K = Kf32.to(torch.float8_e4m3fn)
    if use_v_raw and V_raw is not None:
        V = V_raw
    else:
        V = Vf32.to(torch.float8_e4m3fn)

    # Debug dump: 16 dwords per thread (B regs + A regs)
    threads = (num_q_blocks // 2) * B * H * 512
    debug_mask = int(os.environ.get("NUMERICS_DEBUG_MASK", "0"), 0)
    # Default: 16 dwords per thread (A/B regs). TR8 raw dump needs 32 dwords.
    o_dwords = threads * 16
    if debug_mask & 0x01000000:
        o_dwords = threads * 32
    O = torch.zeros(o_dwords, device="cuda", dtype=torch.uint32)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    debug_mask = int(os.environ.get("NUMERICS_DEBUG_MASK", "0"), 0)
    debug_a = os.environ.get("NUMERICS_DISABLE_A_DEBUG", "0") != "1"
    stride_oh = (s_q * D * 4) | (0x80000000 if debug_a else 0) | debug_mask
    debug_flags = int(os.environ.get("NUMERICS_DEBUG_FLAGS", "0"), 0)
    v_read_cb = int(os.environ.get("NUMERICS_V_READ_CB", "0"), 0)

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

    module, func = load_kernel(
        "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co",
        "_fwd_fp8_scaffold",
    )
    grid = (num_q_blocks // 2, B * H, 1)
    block = (512, 1, 1)
    lds_bytes = 50176

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
    hip.hipModuleUnload(module)

    raw_bytes = O.detach().cpu().view(torch.uint8)
    raw_dump = os.environ.get("NUMERICS_RAW_DUMP", "0")
    if raw_dump == "1":
        out_path = os.environ.get(
            "A_DUMP_OUT",
            "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_a_regs_dump.csv",
        )
        with open(out_path, "w") as f:
            f.write("reg,byte,val_byte\n")
            max_bytes = min(128, raw_bytes.numel())
            for pos in range(max_bytes):
                reg = (pos % 128) // 4
                byte = pos % 4
                val = int(raw_bytes[pos].item())
                f.write(f"{reg},{byte},{val}\n")
        print(f"wrote {out_path}")
        return
    if raw_dump == "2":
        out_path = os.environ.get(
            "A_DUMP_OUT",
            "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_a_regs_dump.csv",
        )
        with open(out_path, "w") as f:
            f.write("pos,val_byte\n")
            max_bytes = min(64, raw_bytes.numel())
            for pos in range(max_bytes):
                val = int(raw_bytes[pos].item())
                f.write(f"{pos},{val}\n")
        print(f"wrote {out_path}")
        return
    if raw_dump == "3":
        out_path = os.environ.get(
            "A_DUMP_OUT",
            "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_a_regs_dump.csv",
        )
        with open(out_path, "w") as f:
            f.write("word,val_u32\n")
            max_bytes = min(48, raw_bytes.numel())
            for i in range(0, max_bytes, 4):
                b0 = int(raw_bytes[i + 0].item())
                b1 = int(raw_bytes[i + 1].item())
                b2 = int(raw_bytes[i + 2].item())
                b3 = int(raw_bytes[i + 3].item())
                val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                f.write(f"{i // 4},{val}\n")
        print(f"wrote {out_path}")
        return
    if raw_dump == "4":
        out_path = os.environ.get(
            "A_DUMP_OUT",
            "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_a_regs_dump.csv",
        )
        bytes_per_lane = 128 if (debug_mask & 0x01000000) else 64
        lanes = min(threads, raw_bytes.numel() // bytes_per_lane)
        with open(out_path, "w") as f:
            f.write("lane,reg,byte,val_byte\n")
            for lane in range(lanes):
                base = lane * bytes_per_lane
                for reg in range(bytes_per_lane // 4):
                    for byte in range(4):
                        pos = base + reg * 4 + byte
                        val = int(raw_bytes[pos].item())
                        f.write(f"{lane},{reg},{byte},{val}\n")
        print(f"wrote {out_path}")
        return

    raw = O.detach().cpu().view(threads, 16)
    bytes_flat = raw.view(torch.uint8).view(threads, 16, 4)
    vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    decoded = vals[bytes_flat.long()].reshape(threads, 64)

    out_path = os.environ.get(
        "A_DUMP_OUT",
        "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_a_regs_dump.csv",
    )
    with open(out_path, "w") as f:
        f.write("lane,bank,reg,byte,val_byte,val_fp8\n")
        for lane in range(min(threads, 512)):
            for reg in range(16):
                for byte in range(4):
                    idx = reg * 4 + byte
                    val = float(decoded[lane, idx].item())
                    bval = int(bytes_flat[lane, reg, byte].item())
                    bank = "B" if reg < 8 else "A"
                    reg_id = reg if reg < 8 else reg - 8
                    f.write(f"{lane},{bank},{reg_id},{byte},{bval},{val}\n")

    print(f"wrote {out_path}")
    for lane in (0, 1, 2, 3, 5, 6, 7):
        if lane >= threads:
            continue
        b_vals = [float(decoded[lane, i].item()) for i in range(32)]
        a_vals = [float(decoded[lane, 32 + i].item()) for i in range(32)]
        print(f"lane {lane} B: {b_vals}")
        print(f"lane {lane} A: {a_vals}")


if __name__ == "__main__":
    main()

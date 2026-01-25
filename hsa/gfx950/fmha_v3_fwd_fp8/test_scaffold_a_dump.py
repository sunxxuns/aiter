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
    Vf32 = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32)
    if V_pattern == "rowid":
        for r in range(s_k):
            Vf32[0, 0, r, :] = float(r)
    else:
        for r in range(s_k):
            Vf32[0, 0, r, :] = float(r)

    Q = Qf32.to(torch.float8_e4m3fn)
    K = Kf32.to(torch.float8_e4m3fn)
    V = Vf32.to(torch.float8_e4m3fn)

    # Debug dump: 16 dwords per thread (B regs + A regs)
    threads = (num_q_blocks // 2) * B * H * 512
    O = torch.zeros(threads * 16, device="cuda", dtype=torch.uint32)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = (s_q * D * 4) | 0x80000000

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
    for lane in (0, 5, 6, 7):
        if lane >= threads:
            continue
        b_vals = [float(decoded[lane, i].item()) for i in range(32)]
        a_vals = [float(decoded[lane, 32 + i].item()) for i in range(32)]
        print(f"lane {lane} B: {b_vals}")
        print(f"lane {lane} A: {a_vals}")


if __name__ == "__main__":
    main()

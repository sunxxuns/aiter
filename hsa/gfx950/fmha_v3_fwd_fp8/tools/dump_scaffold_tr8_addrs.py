#!/usr/bin/env python3
"""
Launch the scaffold kernel in TR8-address-dump mode and write the per-thread
v20..v28 base addresses (plus tid) to a CSV.

This uses the scaffold's debug path (debug_flags 0x02000000) which stores:
  - v20..v23 at byte offset tid*64 + 0
  - v24..v27 at byte offset tid*64 + 16
  - v28, tid, 0, 0 at byte offset tid*64 + 32

Output CSV columns:
  tid, word, val_u32
where word is 0..11 (12 dwords per tid).
"""

from __future__ import annotations

import csv
import ctypes
import os
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")


def load_kernel(co_path: str, kernel_name: str):
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return module, func


def main() -> None:
    # One block (512 threads).
    # Scaffold debug stores use byte offset (tid * 64), i.e. 16 dwords per tid "slot".
    # We write 3x dwordx4 (=12 dwords) into that 16-dword slot.
    out = torch.zeros(512 * 16, device="cuda", dtype=torch.uint32)

    # Minimal dummy inputs (not used by debug path before s_endpgm)
    B, H, D = 1, 1, 128
    S = 64
    num_q_blocks = 2
    num_k_tiles = 2
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32
    q = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = 64  # bytes per tid slot (matches debug stores)
    debug_flags = 0x02000000
    v_read_cb = int(os.environ.get("SCAFFOLD_V_READ_CB", "0"), 0)

    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
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

    hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,  # grid
        512, 1, 1,  # block
        50176,  # LDS bytes
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    host = out.detach().cpu().numpy().astype("uint32").tolist()
    out_path = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/scaffold_tr8_addrs.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tid", "word", "val_u32"])
        for tid in range(512):
            base = tid * 16
            for word in range(16):
                w.writerow([tid, word, int(host[base + word])])
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()


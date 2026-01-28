#!/usr/bin/env python3
"""
Dump the PV B-operand registers (v48..v55) after the pack+mix stage in scaffold.

This uses scaffold debug flag 0x00020000, which stores 8 dwords per thread:
  out[tid, 0..3] = v48..v51
  out[tid, 4..7] = v52..v55
"""

import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import ctypes
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


def main():
    B, H, D = 1, 1, 128
    S = int(os.environ.get("DUMP_S", "64"))
    num_q_blocks = (S + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = (S + 31) // 32
    if num_k_tiles % 2 != 0:
        num_k_tiles += 1
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    torch.manual_seed(0)
    # Use controllable Q/K patterns
    mode = os.environ.get("DUMP_MODE", "ones")
    if mode == "ones":
        Q = torch.ones(B, H, s_q, D, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        K = torch.ones(B, H, s_k, D, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    else:
        Q = (torch.randn(B, H, s_q, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        K = (torch.randn(B, H, s_k, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)

    V = torch.zeros(B, H, s_k, D, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    out = torch.zeros(512 * 8, device="cuda", dtype=torch.uint32)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D
    stride_oh = 32  # bytes per thread slot for dump
    debug_flags = int(os.environ.get("DUMP_FLAGS", "0x00020000"), 0)
    v_read_cb = int(os.environ.get("DUMP_V_READ_CB", "0"), 0)

    args = [
        ctypes.c_void_p(out.data_ptr()),
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
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 512, 1, 1, 50176, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    host = out.detach().cpu().view(512, 8)
    print("tid0 v48..v55:", [hex(int(x)) for x in host[0].tolist()])
    print("tid1 v48..v55:", [hex(int(x)) for x in host[1].tolist()])
    print("tid32 v48..v55:", [hex(int(x)) for x in host[32].tolist()])


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Perf scaffold for QK+PV kernel (no softmax)."""
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import time
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


def main():
    # Benchmark shape
    B, H, S, D = 1, 40, 32130, 128
    num_q_blocks = (S + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = (S + 31) // 32
    S_q = num_q_blocks * 128
    S_k = num_k_tiles * 32

    print("=== QK+PV Scaffold Benchmark ===")
    print(f"B={B}, H={H}, S={S}, D={D}")
    print(f"Q blocks={num_q_blocks}, K tiles={num_k_tiles}")
    print(f"S_q={S_q}, S_k={S_k}")

    # Inputs (FP8)
    torch.manual_seed(0)
    Q = (torch.randn(B, H, S_q, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    K = (torch.randn(B, H, S_k, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    V = (torch.randn(B, H, S_k, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)

    # Output (FP32)
    O = torch.zeros(B, H, S_q, D, device="cuda", dtype=torch.float32)

    stride_qh = S_q * D        # bytes for FP8
    stride_kh = S_k * D
    stride_vh = S_k * D
    stride_oh = S_q * D * 4    # bytes for FP32

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
    waves_per_block = block[0] // 64
    blocks = grid[0] * grid[1] * grid[2]
    print(f"grid={grid}, block={block}, waves_per_block={waves_per_block}, blocks={blocks}")

    # Warmup
    for _ in range(3):
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

    # Timed runs
    iters = 5
    t0 = time.time()
    for _ in range(iters):
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
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / iters

    # FLOPs (equivalent full attention, for comparison to target)
    eq_flops = 4.0 * B * H * S * S * D
    eq_tflops = eq_flops / (avg_ms / 1000.0) / 1.0e12

    # FLOPs actually executed by this scaffold
    pv_k = int(os.environ.get("SCAFFOLD_PV_K", "64"))
    pv_mfma = int(os.environ.get("SCAFFOLD_PV_MFMA", "4"))
    qk_flops = 2.0 * 32 * 32 * 64 * 2  # 2× MFMA K=64
    pv_flops = pv_mfma * 2.0 * 32 * 32 * pv_k
    waves_per_block = block[0] // 64
    blocks = grid[0] * grid[1] * grid[2]
    exec_flops = blocks * num_k_tiles * waves_per_block * (qk_flops + pv_flops)
    exec_tflops = exec_flops / (avg_ms / 1000.0) / 1.0e12

    print(f"avg_ms: {avg_ms:.3f}")
    print(f"TF/s (equivalent full-attention): {eq_tflops:.1f}")
    print(f"TF/s (executed MFMA): {exec_tflops:.1f}")


if __name__ == "__main__":
    main()

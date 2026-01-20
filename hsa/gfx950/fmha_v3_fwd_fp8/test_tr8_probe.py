#!/usr/bin/env python3
"""Probe TR8 interleaved layout mapping."""
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import ctypes
import subprocess
import numpy as np
import torch

hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_tr8_probe.s"
    obj = f"{base_dir}/fwd_fp8_tr8_probe.o"
    co = f"{base_dir}/fwd_fp8_tr8_probe.co"
    cmd = (
        f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa "
        f"-mcpu=gfx950 -mno-xnack -c {src} -o {obj} && "
        f"/opt/rocm/llvm/bin/ld.lld -shared {obj} -o {co}"
    )
    subprocess.check_call(cmd, shell=True)
    return co


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
    print("=== TR8 Probe ===")
    co = build_kernel()
    module, func = load_kernel(co, "_fwd_fp8_tr8_probe")

    # Input Q[32x128] with value = encoded (addr >> 3)
    # Interleaved layout address: addr = (row % 8) + (row / 8) * 1024 + k * 8
    rows, cols = 32, 128
    q_data = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for k in range(cols):
            addr = (r % 8) + (r // 8) * 1024 + k * 8
            val = (addr >> 3) & 0x7F
            val |= ((addr >> 11) & 1) << 7
            q_data[r, k] = val

    q_input = torch.from_numpy(q_data.flatten()).to(device="cuda")
    output = torch.zeros(512, dtype=torch.uint8, device="cuda")

    base_offsets = [0, 2, 4, 6, 8, 64, 128, 512, 1024, 1088, 4096, 5184]
    results = {}
    for base_offset in base_offsets:
        output.zero_()
        args = [
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(q_input.data_ptr()),
            ctypes.c_int32(base_offset),
        ]
        args_ptrs = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
        )

        hip.hipModuleLaunchKernel(func, 1, 1, 1, 256, 1, 1, 8192, None, args_ptrs, None)
        hip.hipDeviceSynchronize()

        out = output.cpu().numpy()
        results[base_offset] = list(out[0:8])  # thread 0 only

    print("\nThread 0 bytes for selected base offsets:")
    for base_offset in base_offsets:
        print(f"{base_offset:04d}: {results[base_offset]}")

    hip.hipModuleUnload(module)


if __name__ == "__main__":
    main()

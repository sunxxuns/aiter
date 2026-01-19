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

    # Input Q[8x128] with Q[r,k] = r*128 + k (mod 256)
    rows, cols = 8, 128
    q_data = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for k in range(cols):
            q_data[r, k] = (r * cols + k) % 256

    q_input = torch.from_numpy(q_data.flatten()).to(device="cuda")
    output = torch.zeros(512, dtype=torch.uint8, device="cuda")

    for base_offset in range(0, 8):
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
        print(f"\nBase offset = {base_offset}")
        for tid in range(4):
            row = tid % 8
            expected = [q_data[(row + base_offset) % rows, k] for k in range(8)]
            actual = list(out[tid * 8 : tid * 8 + 8])
            print(f"  Thread {tid} (row={row}): expected={expected}, got={actual}")

    hip.hipModuleUnload(module)


if __name__ == "__main__":
    main()

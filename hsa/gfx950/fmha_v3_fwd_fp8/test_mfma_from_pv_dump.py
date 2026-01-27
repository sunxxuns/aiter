#!/usr/bin/env python3
"""Run MFMA on dumped PV A/B regs to validate mapping."""
import ctypes
import subprocess
import torch

BASE_DIR = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
SRC = f"{BASE_DIR}/fwd_fp8_mfma_map_debug.s"
CO = f"{BASE_DIR}/fwd_fp8_mfma_map_debug.co"
DUMP = f"{BASE_DIR}/p_a_regs_dump.csv"


def build_kernel():
    obj = f"{BASE_DIR}/fwd_fp8_mfma_map_debug.o"
    cmd = (
        f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa "
        f"-mcpu=gfx950 -mno-xnack -c {SRC} -o {obj} && "
        f"/opt/rocm/llvm/bin/ld.lld -shared {obj} -o {CO}"
    )
    subprocess.check_call(cmd, shell=True)
    return CO


def load_kernel(co_path, kernel_name):
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    ret = hip.hipModuleLoad(ctypes.byref(module), co_path.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleLoad failed: {ret}")
    ret = hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if ret != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {ret}")
    return hip, module, func


def build_byte_lut():
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    lut = {}
    for i, v in enumerate(table):
        val = float(v.item())
        if val not in lut:
            lut[val] = i
    return lut


def main():
    # Build packed regs from CSV
    byte_lut = build_byte_lut()
    packed = torch.zeros(64, 16, dtype=torch.uint32)
    bytes_flat = torch.zeros(64, 16, 4, dtype=torch.uint8)
    import csv
    with open(DUMP, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lane = int(row["lane"])
            if lane >= 64:
                continue
            reg = int(row["reg"])
            byte = int(row["byte"])
            bank = row["bank"]
            if bank == "B":
                reg += 8
            if "val_byte" in row and row["val_byte"] != "":
                b = int(row["val_byte"])
            else:
                val = float(row["val_fp8"])
                b = byte_lut.get(val, 0)
            bytes_flat[lane, reg, byte] = b
    packed = bytes_flat.view(torch.uint32)

    co = build_kernel()
    hip, module, func = load_kernel(co, "_fwd_fp8_mfma_map_debug")

    in_buf = packed.to(device="cuda")
    out = torch.zeros(64 * 16, device="cuda", dtype=torch.float32)

    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(in_buf.data_ptr()),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,
        64, 1, 1,
        0,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    raw = out.detach().cpu().view(64, 16)
    decoded = torch.empty((32, 32), dtype=torch.float32)
    for tid in range(64):
        lane = tid & 63
        col = lane & 31
        row_base = ((lane >> 5) & 1) * 4
        for i in range(16):
            row = (i % 4) + row_base + (i // 4) * 8
            decoded[row, col] = raw[tid, i]

    print("mfma decoded[0:8,0:8]:")
    print(decoded[:8, :8])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Digit-probe mapping for mixed P pack (post-mix)."""
import ctypes
import subprocess
import torch

torch.manual_seed(0)
os_env = __import__("os").environ
os_env["HIP_VISIBLE_DEVICES"] = "0"
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")

BASE_DIR = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
SRC = f"{BASE_DIR}/fwd_fp8_p_pack_dump.s"
CO = f"{BASE_DIR}/fwd_fp8_p_pack_dump.co"


def build_kernel():
    obj = f"{BASE_DIR}/fwd_fp8_p_pack_dump.o"
    cmd = (
        f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa "
        f"-mcpu=gfx950 -mno-xnack -c {SRC} -o {obj} && "
        f"/opt/rocm/llvm/bin/ld.lld -shared {obj} -o {CO}"
    )
    subprocess.check_call(cmd, shell=True)
    return CO


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


def run_dump(mode, digit_idx):
    B, H, D = 1, 1, 128
    Q_rows = 128
    K_rows = 64
    dtype = torch.float8_e4m3fn

    Q_f32 = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
    K_f32 = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)

    for r in range(32):
        Q_f32[0, 0, r, r] = 1.0

    if mode == "row":
        for k in range(K_rows):
            for r in range(32):
                digit = (r >> (digit_idx * 2)) & 0x3
                if digit_idx == 2:
                    digit = (r >> 4) & 0x1
                K_f32[0, 0, k, r] = float(digit)
    else:
        for k in range(K_rows):
            for r in range(32):
                digit = (k >> (digit_idx * 2)) & 0x3
                K_f32[0, 0, k, r] = float(digit)

    Q = Q_f32.to(dtype)
    K = K_f32.to(dtype)

    out = torch.empty(256 * 8, dtype=torch.uint32, device="cuda")

    co_path = build_kernel()
    module, func = load_kernel(co_path, "_fwd_fp8_p_pack_dump")

    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,
        256, 1, 1,
        24576,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    raw = out.detach().cpu().view(256, 8)
    bytes_flat = raw.view(torch.uint8).view(256, 8, 4)
    vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    decoded = vals[bytes_flat.long()].reshape(256, 32)
    digits = torch.round(decoded).clamp(0, 3).to(torch.int32)
    return digits


def main():
    row_d0 = run_dump("row", 0)
    row_d1 = run_dump("row", 1)
    row_d2 = run_dump("row", 2)

    k_d0 = run_dump("k", 0)
    k_d1 = run_dump("k", 1)
    k_d2 = run_dump("k", 2)

    row_val = row_d0 + (row_d1 * 4) + (row_d2 * 16)
    k_val = k_d0 + (k_d1 * 4) + (k_d2 * 16)

    out_path = f"{BASE_DIR}/p_pack_mapping_digits.csv"
    with open(out_path, "w") as f:
        f.write("lane,pos,row,k\n")
        for lane in range(64):
            for pos in range(32):
                f.write(f"{lane},{pos},{int(row_val[lane, pos])},{int(k_val[lane, pos])}\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

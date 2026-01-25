#!/usr/bin/env python3
"""Product-code mapping for mixed P pack (post-mix)."""
import ctypes
import random
import subprocess
import torch

torch.manual_seed(0)
random.seed(0)
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


def run_dump(row_code, k_code):
    B, H, D = 1, 1, 128
    Q_rows = 128
    K_rows = 64
    dtype = torch.float8_e4m3fn

    Q_f32 = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
    K_f32 = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)

    for r in range(32):
        Q_f32[0, 0, r, r] = float(row_code[r])
    for k in range(K_rows):
        for r in range(32):
            K_f32[0, 0, k, r] = float(k_code[k])

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
    return decoded


def build_codes(seed):
    rng = random.Random(seed)
    row_code = [rng.randint(1, 16) for _ in range(32)]
    k_code = [rng.randint(1, 16) for _ in range(64)]
    return row_code, k_code


def main():
    runs = []
    for seed in (1, 2, 3, 4):
        row_code, k_code = build_codes(seed)
        decoded = run_dump(row_code, k_code)
        runs.append((row_code, k_code, decoded))

    # Build lookup for (row,k) -> triple
    lookup = {}
    for r in range(32):
        for k in range(64):
            triple = []
            for row_code, k_code, _ in runs:
                triple.append(row_code[r] * k_code[k])
            lookup[tuple(triple)] = (r, k)

    out_path = f"{BASE_DIR}/p_pack_mapping_products.csv"
    with open(out_path, "w") as f:
        f.write("lane,pos,row,k\n")
        for lane in range(64):
            for pos in range(32):
                triple = tuple(int(round(float(run[2][lane, pos].item()))) for run in runs)
                row_k = lookup.get(triple, (-1, -1))
                f.write(f"{lane},{pos},{row_k[0]},{row_k[1]}\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

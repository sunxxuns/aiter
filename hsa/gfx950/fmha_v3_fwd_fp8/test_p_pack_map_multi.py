#!/usr/bin/env python3
"""Multi-run mapping for mixed P pack using product codes."""
import ctypes
import os
import random
import subprocess
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"
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


def fp8_table():
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = table[torch.isfinite(table)]
    finite = finite[finite != 0]
    return finite


def run_dump(Q, K):
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


def build_codes(rng, table, size):
    idx = rng.sample(range(table.numel()), size)
    return table[idx].clone()


def quantize_fp8(values):
    return values.to(torch.float8_e4m3fn).float()


def main():
    runs = int(os.environ.get("P_PACK_RUNS", "6"))
    rng = random.Random(0)
    table = fp8_table()

    collected = []
    for i in range(runs):
        row_code = build_codes(rng, table, 32)
        k_code = build_codes(rng, table, 64)
        Qf32 = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.float32)
        Kf32 = torch.zeros(1, 1, 64, 128, device="cuda", dtype=torch.float32)
        for r in range(32):
            Qf32[0, 0, r, r] = row_code[r]
        for k in range(64):
            for r in range(32):
                Kf32[0, 0, k, r] = k_code[k]
        Q = Qf32.to(torch.float8_e4m3fn)
        K = Kf32.to(torch.float8_e4m3fn)
        decoded = run_dump(Q, K)
        collected.append((row_code, k_code, decoded))

    # Build expected tuple for each (row,k)
    expected = {}
    for r in range(32):
        for k in range(64):
            tup = []
            for row_code, k_code, _ in collected:
                val = quantize_fp8(row_code[r] * k_code[k]).item()
                tup.append(float(val))
            expected[tuple(tup)] = (r, k)

    out_path = f"{BASE_DIR}/p_pack_mapping_multi.csv"
    missing = 0
    with open(out_path, "w") as f:
        f.write("lane,pos,row,k\n")
        for lane in range(64):
            for pos in range(32):
                tup = []
                for _, _, decoded in collected:
                    tup.append(float(decoded[lane, pos].item()))
                row_k = expected.get(tuple(tup))
                if row_k is None:
                    missing += 1
                    f.write(f"{lane},{pos},-1,-1\n")
                else:
                    f.write(f"{lane},{pos},{row_k[0]},{row_k[1]}\n")
    print(f"wrote {out_path} missing={missing}")


if __name__ == "__main__":
    main()

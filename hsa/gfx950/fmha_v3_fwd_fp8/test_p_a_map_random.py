#!/usr/bin/env python3
"""Random-probe mapping for P->A transpose (post-transpose)."""
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


def build_kernel():
    obj = f"{BASE_DIR}/fwd_fp8_p_pack_dump.o"
    co = f"{BASE_DIR}/fwd_fp8_p_pack_dump.co"
    cmd = (
        f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa "
        f"-mcpu=gfx950 -mno-xnack -c {SRC} -o {obj} && "
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


def build_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    B, H, D = 1, 1, 128
    Q_rows = 128
    K_rows = 32
    Qf = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
    Kf = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)
    Qf[0, 0, :32, :] = (torch.randn(32, D, device="cuda") * 0.5).float()
    Kf[0, 0, :32, :] = (torch.randn(32, D, device="cuda") * 0.5).float()
    return Qf, Kf


def main():
    runs = []
    for seed in (1, 2, 3):
        Qf, Kf = build_random(seed)
        Q = Qf.to(torch.float8_e4m3fn)
        K = Kf.to(torch.float8_e4m3fn)
        P = torch.matmul(Q.float(), K.float().transpose(-1, -2)).to(torch.float8_e4m3fn).float()
        dumped = run_dump(Q, K)
        runs.append((P[0, 0].detach().cpu(), dumped))

    out_path = f"{BASE_DIR}/p_a_mapping_random.csv"
    with open(out_path, "w") as f:
        f.write("lane,pos,row,k\n")
        for lane in range(64):
            for pos in range(32):
                candidates = None
                for P, dumped in runs:
                    val = float(dumped[lane, pos].item())
                    matches = {(r, k) for r in range(32) for k in range(32) if float(P[r, k].item()) == val}
                    candidates = matches if candidates is None else candidates & matches
                if not candidates:
                    f.write(f"{lane},{pos},-1,-1\n")
                else:
                    row, k = sorted(candidates)[0]
                    f.write(f"{lane},{pos},{row},{k}\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

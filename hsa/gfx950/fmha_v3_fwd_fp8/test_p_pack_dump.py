#!/usr/bin/env python3
"""Dump packed P->A operand bytes for layout debugging."""
import ctypes
import os
import subprocess
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_p_pack_dump.s"
    obj = f"{base_dir}/fwd_fp8_p_pack_dump.o"
    co = f"{base_dir}/fwd_fp8_p_pack_dump.co"
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


def decode_fp8_bytes(byte_tensor):
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return table[byte_tensor.long()]


def main():
    print("=== P Pack Dump ===")
    B, H, D = 1, 1, 128
    Q_rows = 128
    K_rows = 32

    dtype = torch.float8_e4m3fn
    pattern = os.environ.get("P_PACK_PATTERN", "k")

    Q_f32 = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
    K_f32 = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)

    if pattern == "identity":
        for r in range(K_rows):
            Q_f32[0, 0, r, r] = 1.0
            K_f32[0, 0, r, r] = 1.0
    elif pattern == "rowdiag":
        for r in range(K_rows):
            Q_f32[0, 0, r, r] = float(r) / 64.0
            K_f32[0, 0, r, r] = 1.0
    else:
        # Q identity (rows 0..31)
        for r in range(K_rows):
            Q_f32[0, 0, r, r] = 1.0
        # K rows encode their row id (K[k, :] = k)
        for k in range(K_rows):
            K_f32[0, 0, k, :] = float(k)

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
    decoded = decode_fp8_bytes(bytes_flat)

    for tid in [0, 1, 2, 3, 16, 17, 32, 33]:
        vals = decoded[tid].reshape(-1)
        if pattern == "rowdiag":
            nz = (vals.abs() > 1e-4).nonzero().view(-1)
            nz_idx = nz.tolist()
            nz_vals = vals[nz].tolist()
            print(f"tid {tid} nz idx:", nz_idx)
            print(f"tid {tid} nz val:", [round(v, 4) for v in nz_vals])
        else:
            k_map = vals.round().to(torch.int32).tolist()
            print(f"tid {tid} k-map:", k_map)


if __name__ == "__main__":
    main()

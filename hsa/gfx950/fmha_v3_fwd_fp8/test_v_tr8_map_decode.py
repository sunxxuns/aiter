#!/usr/bin/env python3
"""Decode TR8 reads into row/col mapping."""
import ctypes
import os
import subprocess
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_v_tr8_debug.s"
    obj = f"{base_dir}/fwd_fp8_v_tr8_debug.o"
    co = f"{base_dir}/fwd_fp8_v_tr8_debug.co"
    cmd = (
        f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa "
        f"-mcpu=gfx950 -mno-xnack -c {src} -o {obj} && "
        f"/opt/rocm/llvm/bin/ld.lld -shared {obj} -o {co}"
    )
    subprocess.check_call(cmd, shell=True)
    return co


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


def fill_v_pattern(v_bytes, pattern):
    s_k, d = v_bytes.shape[-2], v_bytes.shape[-1]
    if pattern == "row":
        for r in range(s_k):
            v_bytes[0, 0, r, :] = r & 0xFF
    elif pattern == "col":
        for c in range(d):
            v_bytes[0, 0, :, c] = c & 0xFF
    elif pattern == "rowcol2":
        for r in range(s_k):
            for c in range(d):
                v_bytes[0, 0, r, c] = ((r & 0x3F) << 2) | (c & 0x3)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def run_pattern(pattern):
    torch.manual_seed(0)
    torch.cuda.init()
    co = build_kernel()

    s = 64
    num_q_blocks = (s + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = (s + 31) // 32
    if num_k_tiles % 2 != 0:
        num_k_tiles += 1
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32
    d = 128

    v = torch.empty((1, 1, s_k, d), device="cuda", dtype=torch.float8_e4m3fn)
    v_bytes = v.view(torch.uint8)
    fill_v_pattern(v_bytes, pattern)

    q = torch.zeros((1, 1, s_q, d), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros((1, 1, s_k, d), device="cuda", dtype=torch.float8_e4m3fn)

    out = torch.zeros(256 * 32, device="cuda", dtype=torch.uint32)

    hip, module, func = load_kernel(co, "_fwd_fp8_v_tr8_debug")
    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_int32(num_k_tiles),
        ctypes.c_int32(s_q * d),
        ctypes.c_int32(s_k * d),
        ctypes.c_int32(s_k * d),
        ctypes.c_int32(s_q * d * 4),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,
        256, 1, 1,
        0,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    out_cpu = out.detach().cpu().numpy().astype("uint32")
    bytes_out = []
    for tid in range(256):
        words = out_cpu[tid * 32:(tid + 1) * 32]
        bytes_list = []
        for w in words:
            bytes_list.extend(list(int(w).to_bytes(4, "little")))
        bytes_out.append(bytes_list)
    return bytes_out


def main():
    rows = run_pattern("row")
    cols = run_pattern("col")
    rc2 = run_pattern("rowcol2")

    tids = [0, 1, 2, 3, 8, 9, 10, 11, 32, 33, 34, 35]
    for tid in tids:
        print(f"\n=== tid {tid} ===")
        for chunk in range(16):
            base = chunk * 8
            row_vals = rows[tid][base:base + 8]
            col_vals = cols[tid][base:base + 8]
            rc2_vals = rc2[tid][base:base + 8]
            row_dec = [int(v) for v in row_vals]
            col_dec = [int(v) for v in col_vals]
            rc2_row = [int(v) >> 2 for v in rc2_vals]
            rc2_col = [int(v) & 0x3 for v in rc2_vals]
            print(f"chunk {chunk:02d} rows {row_dec} cols {col_dec} rc2_row {rc2_row} rc2_col {rc2_col}")


if __name__ == "__main__":
    main()

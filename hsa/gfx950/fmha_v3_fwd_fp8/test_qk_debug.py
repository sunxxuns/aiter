#!/usr/bin/env python3
"""Numerics check for QK debug kernel (single K tile)."""
import os
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import ctypes
import math
import subprocess
import torch

hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_qk_debug.s"
    obj = f"{base_dir}/fwd_fp8_qk_debug.o"
    co = f"{base_dir}/fwd_fp8_qk_debug.co"
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


def decode_qk(raw, rows, cols):
    """Decode MFMA v32-v47 layout to row-major (rows x cols)."""
    decoded = torch.empty((rows, cols), dtype=torch.float32)
    raw_mat = raw.view(256, 16)
    use_f = os.environ.get("DEBUG_DECODE_F", "0") == "1"
    for tid in range(256):
        lane = tid & 63
        wave = tid >> 6
        col = (lane | (lane >> 1)) & 31 if use_f else (lane & 31)
        lane_hi = lane >> 5
        for i in range(16):
            row = (i % 4) + lane_hi * 4 + (i // 4) * 8
            row_global = wave * 32 + row
            decoded[row_global, col] = raw_mat[tid, i]
    return decoded


def main():
    print("=== QK Debug Numerics ===")
    B, H, D = 1, 1, 128
    Q_rows = 128
    K_rows = 32

    dtype = torch.float8_e4m3fn
    torch.manual_seed(0)
    if os.environ.get("DEBUG_RANDOM_NONUNIFORM", "0") == "1":
        Q_f32 = (torch.randn(B, H, Q_rows, D, device="cuda") * 0.05 + 0.02)
        K_f32 = (torch.randn(B, H, K_rows, D, device="cuda") * 0.05 + 0.02)
        q_row_scale = torch.linspace(0.6, 1.4, Q_rows, device="cuda").view(1, 1, Q_rows, 1)
        q_col_scale = torch.linspace(0.7, 1.3, D, device="cuda").view(1, 1, 1, D)
        k_row_scale = torch.linspace(1.3, 0.7, K_rows, device="cuda").view(1, 1, K_rows, 1)
        k_col_scale = torch.linspace(0.8, 1.2, D, device="cuda").view(1, 1, 1, D)
        Q_f32 = Q_f32 * q_row_scale * q_col_scale
        K_f32 = K_f32 * k_row_scale * k_col_scale
        Q = Q_f32.to(dtype)
        K = K_f32.to(dtype)
    elif os.environ.get("DEBUG_COLMAP", "0") == "1":
        Q_f32 = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
        K_f32 = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)
        Q_f32[0, 0, 0, 0] = 1.0
        for r in range(K_rows):
            K_f32[0, 0, r, 0] = (r + 1) / 64.0
        Q = Q_f32.to(dtype)
        K = K_f32.to(dtype)
    elif os.environ.get("DEBUG_IDENTITY", "0") == "1":
        Q_f32 = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
        K_f32 = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)
        for i in range(Q_rows):
            Q_f32[0, 0, i, i] = 1.0
        for i in range(K_rows):
            K_f32[0, 0, i, i] = 1.0
        Q = Q_f32.to(dtype)
        K = K_f32.to(dtype)
    else:
        Q = (torch.randn(B, H, Q_rows, D, device="cuda") * 0.1).to(dtype)
        K = (torch.randn(B, H, K_rows, D, device="cuda") * 0.1).to(dtype)
    if os.environ.get("DEBUG_ZERO_Q", "0") == "1":
        Q.zero_()
    if os.environ.get("DEBUG_ZERO_K", "0") == "1":
        K.zero_()

    print(f"Q finite: {torch.isfinite(Q.float()).all().item()}")
    print(f"K finite: {torch.isfinite(K.float()).all().item()}")

    O = torch.zeros(256 * 16, dtype=torch.float32, device="cuda")

    co_path = build_kernel()
    module, func = load_kernel(co_path, "_fwd_fp8_qk_debug")

    args = [
        ctypes.c_void_p(O.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    grid = (1, 1, 1)
    block = (256, 1, 1)
    lds_bytes = 24576
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
    hip.hipModuleUnload(module)

    raw = O.detach().cpu()
    print(f"kernel output finite: {torch.isfinite(raw).all().item()}")

    decoded = decode_qk(raw, Q_rows, K_rows)
    Qf = Q.float()[0, 0]
    Kf = K.float()[0, 0]
    ref = torch.matmul(Qf, Kf.transpose(0, 1)).cpu()

    diff = (decoded - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (ref.abs() + 1e-6)).max().item()
    print(f"max_err: {max_err:.6f}")
    print(f"mean_err: {mean_err:.6f}")
    print(f"max_rel_err: {rel_err:.6f}")
    print(f"decoded finite: {torch.isfinite(decoded).all().item()}")

    if os.environ.get("DEBUG_COLMAP", "0") == "1":
        row0 = decoded[0, :K_rows]
        expected = torch.tensor([(r + 1) / 64.0 for r in range(K_rows)])
        print("row0 decoded:")
        print(row0)
        print("expected:")
        print(expected)
        mapped = torch.round(row0 * 64 - 1).to(torch.int32)
        print("mapped rows:")
        print(mapped)
    elif os.environ.get("DEBUG_IDENTITY", "0") == "1":
        diag = torch.diagonal(decoded[:K_rows, :K_rows]).cpu()
        off_diag = decoded[:K_rows, :K_rows] - torch.diag(diag)
        print(f"diag min/max: {diag.min().item():.3f}/{diag.max().item():.3f}")
        print(f"off-diag max: {off_diag.abs().max().item():.3f}")
        ref_t = ref.transpose(0, 1)
        diff_ref = (decoded - ref).abs().mean().item()
        diff_ref_t = (decoded.transpose(0, 1) - ref_t).abs().mean().item()
        print(f"mean_err vs ref: {diff_ref:.6f}, vs ref^T (decoded.T): {diff_ref_t:.6f}")
        print("decoded[0:8,0:8]:")
        print(decoded[:8, :8])
        print("ref[0:8,0:8]:")
        print(ref[:8, :8])
        nz = (decoded[:K_rows, :K_rows].abs() > 0.5).nonzero()
        if nz.numel() > 0:
            print("first non-zero positions (row,col,val):")
            for idx in nz[:10]:
                r, c = idx.tolist()
                print(r, c, decoded[r, c].item())
    elif os.environ.get("DEBUG_RANDOM_NONUNIFORM", "0") == "1":
        flat_dec = decoded.flatten()
        flat_ref = ref.flatten()
        corr = torch.corrcoef(torch.stack([flat_dec, flat_ref]))[0, 1].item()
        max_thr = float(os.environ.get("DEBUG_RANDOM_MAX_THR", "1.0"))
        mean_thr = float(os.environ.get("DEBUG_RANDOM_MEAN_THR", "0.05"))
        corr_thr = float(os.environ.get("DEBUG_RANDOM_CORR_THR", "0.95"))
        passed = (max_err <= max_thr) and (mean_err <= mean_thr) and (corr >= corr_thr)
        print(f"corr: {corr:.6f}")
        print(f"PASS: {passed} (max<= {max_thr}, mean<= {mean_thr}, corr>= {corr_thr})")


if __name__ == "__main__":
    main()

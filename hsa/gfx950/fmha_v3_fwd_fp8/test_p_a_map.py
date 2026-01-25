#!/usr/bin/env python3
"""Map packed P -> A operand bytes (post-transpose)."""
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


def build_codebook(count=32):
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = table[torch.isfinite(table)]
    finite = finite[finite > 0]
    finite = finite[(finite >= 1.0) & (finite <= 16.0)]
    unique = torch.unique(finite)
    unique, _ = torch.sort(unique)
    if unique.numel() < count:
        raise RuntimeError(f"Not enough FP8 codes: {unique.numel()}")
    idx = torch.linspace(0, unique.numel() - 1, steps=count).round().long()
    return unique[idx]


def map_to_codes(values, codes):
    codes_f = codes.view(1, 1, -1)
    vals = values.unsqueeze(-1)
    diff = (vals - codes_f).abs()
    idx = diff.argmin(dim=-1)
    return idx.to(torch.int32)


def run_dump(pattern, codes=None):
    B, H, D = 1, 1, 128
    Q_rows = 128
    K_rows = int(os.environ.get("P_A_K_ROWS", "32"))
    tile = int(os.environ.get("P_A_TILE", "0"))
    dtype = torch.float8_e4m3fn

    Q_f32 = torch.zeros(B, H, Q_rows, D, device="cuda", dtype=torch.float32)
    K_f32 = torch.zeros(B, H, K_rows, D, device="cuda", dtype=torch.float32)

    if pattern in ("k_code", "row_code"):
        if codes is None:
            raise RuntimeError("codes required for k_code/row_code")
        codes_f = codes.to(device="cuda", dtype=torch.float32)
        Q_f32[0, 0, :K_rows, :K_rows] = torch.eye(K_rows, device="cuda")
        if tile == 1 and int(os.environ.get("P_A_Q_SHIFT", "1")):
            Q_f32.zero_()
            for r in range(32):
                Q_f32[0, 0, r, 32 + r] = 1.0
        if pattern == "k_code":
            if tile == 0:
                K_f32[0, 0, :32, :32] = codes_f.view(32, 1).expand(32, 32)
            elif tile == 1:
                K_f32[0, 0, 32:64, :32] = codes_f.view(32, 1).expand(32, 32)
            else:
                K_f32[0, 0, :32, :32] = codes_f.view(32, 1).expand(32, 32)
                K_f32[0, 0, 32:64, :32] = codes_f.view(32, 1).expand(32, 32)
        else:
            if tile == 0:
                K_f32[0, 0, :32, :32] = codes_f.view(1, 32).expand(32, 32)
            elif tile == 1:
                K_f32[0, 0, 32:64, :32] = codes_f.view(1, 32).expand(32, 32)
            else:
                K_f32[0, 0, :32, :32] = codes_f.view(1, 32).expand(32, 32)
                K_f32[0, 0, 32:64, :32] = codes_f.view(1, 32).expand(32, 32)
    else:
        for r in range(K_rows):
            Q_f32[0, 0, r, r] = 1.0
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
    decoded = decode_fp8_bytes(bytes_flat).reshape(256, 32)
    return decoded


def main():
    codes = build_codebook(count=32)
    k_vals = run_dump("k_code", codes=codes)
    row_vals = run_dump("row_code", codes=codes)

    k_idx = map_to_codes(k_vals, codes)
    pack_tile = int(os.environ.get("P_A_TILE", "0"))
    if pack_tile == 1:
        k_idx = k_idx + 32
    elif pack_tile > 1:
        lane_ids = (torch.arange(k_idx.shape[0]).view(-1, 1) % 64)
        k_idx = k_idx + ((lane_ids >= 32).to(torch.int32) * 32)
    row_idx = map_to_codes(row_vals, codes)

    # Dump a CSV for full mapping (allow missing entries)
    out_path = os.environ.get(
        "P_A_OUT",
        "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_a_mapping.csv",
    )
    if pack_tile > 1:
        k_base = 0
        k_span = 64
    else:
        k_base = 32 * pack_tile
        k_span = 32
    missing = 0
    with open(out_path, "w") as f:
        f.write("row,k,dst_lane,dst_reg,dst_byte,src_lane,src_reg,src_byte\n")
        for row in range(32):
            for k in range(k_base, k_base + k_span):
                dst_lane = row
                dst_reg = (k - k_base) // 4
                dst_byte = (k - k_base) % 4
                # find source in decoded output
                src_lane = -1
                src_reg = -1
                src_byte = -1
                for lane in range(64):
                    for pos in range(32):
                        r = int(row_idx[lane, pos].item())
                        kk = int(k_idx[lane, pos].item())
                        if r == row and kk == k:
                            src_lane = lane
                            src_reg = pos // 4
                            src_byte = pos % 4
                            break
                    if src_lane >= 0:
                        break
                if src_lane < 0:
                    missing += 1
                f.write(f"{row},{k},{dst_lane},{dst_reg},{dst_byte},{src_lane},{src_reg},{src_byte}\n")
    if missing:
        print(f"missing mappings: {missing}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

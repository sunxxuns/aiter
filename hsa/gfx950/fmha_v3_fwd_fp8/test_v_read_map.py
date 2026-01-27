#!/usr/bin/env python3
"""Dump PV A-operand reads (V LDS) and map to row ids."""
import ctypes
import os
import subprocess
import torch

torch.manual_seed(0)
os_env = __import__("os").environ
os_env["HIP_VISIBLE_DEVICES"] = "0"
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
_ = torch.zeros(1, device="cuda")

BASE_DIR = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
SRC = f"{BASE_DIR}/fwd_fp8_v_read_dump.s"
CO = f"{BASE_DIR}/fwd_fp8_v_read_dump.co"


def build_kernel():
    obj = f"{BASE_DIR}/fwd_fp8_v_read_dump.o"
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


def build_codebook(count=32):
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = table[torch.isfinite(table)]
    finite = finite[finite > 0]
    finite = finite[finite <= 16.0]
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


def decode_fp8_bytes(byte_tensor):
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return table[byte_tensor.long()]


def main():
    # Build V with per-row or per-col code values
    codes = build_codebook(32)
    codes_f = codes.to(device="cuda", dtype=torch.float32)
    B, H, D = 1, 1, 128
    S_k = int(os_env.get("V_READ_SK", "128"))
    V = torch.zeros(B, H, S_k, D, device="cuda", dtype=torch.float32)
    V_raw = None
    v_map = os_env.get("V_READ_MAP", "row")
    tile = int(os_env.get("V_READ_TILE", "0"))
    use_v_raw = os.environ.get("V_READ_RAW", "0") == "1"
    if v_map == "col":
        for c in range(32):
            V[0, 0, :, c] = codes_f[c]
            V[0, 0, :, 32 + c] = codes_f[c]
    elif v_map == "blockk":
        col_bytes = torch.arange(128, dtype=torch.long, device="cuda")
        col_blocks = (col_bytes // 32) & 3
        if use_v_raw:
            v_bytes = torch.empty((S_k, 128), dtype=torch.uint8, device="cuda")
            for r in range(S_k):
                k_id = r & 31
                byte_ids = ((col_blocks << 5) | k_id).to(torch.uint8)
                v_bytes[r, :] = byte_ids
            V_raw = v_bytes.view(torch.float8_e4m3fn)
            V[0, 0, :, :] = V_raw.float()
        else:
            byte_lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device="cuda")
            for r in range(S_k):
                k_id = r & 31
                byte_ids = ((col_blocks << 5) | k_id).to(torch.long)
                V[0, 0, r, :] = byte_lut[byte_ids].float()
    elif v_map == "colbyte":
        byte_lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device="cuda")
        col_bytes = torch.arange(128, dtype=torch.long, device="cuda")
        col_vals = byte_lut[col_bytes].float()
        V[0, 0, :, :] = col_vals
    elif v_map == "colblock":
        block_codes = codes_f[:4]
        for b in range(4):
            c0 = b * 32
            c1 = c0 + 32
            V[0, 0, :, c0:c1] = float(block_codes[b].item())
    else:
        for r in range(S_k):
            V[0, 0, r, :] = codes_f[r % 32]
    if use_v_raw and V_raw is not None:
        V = V_raw
    else:
        V = V.to(torch.float8_e4m3fn)

    out = torch.zeros(256 * 40, device="cuda", dtype=torch.uint32)

    co = build_kernel()
    module, func = load_kernel(co, "_fwd_fp8_v_read_dump")

    stride_vh = ctypes.c_int32(S_k * D)
    v_read_offset = ctypes.c_int32(int(os_env.get("V_READ_OFFSET", "0")))
    v_read_lane_xor = ctypes.c_int32(int(os_env.get("V_READ_LANE_XOR", "0"), 0))
    v_read_base_xor = ctypes.c_int32(int(os_env.get("V_READ_BASE_XOR", "0"), 0))
    v_read_s25_xor = ctypes.c_int32(int(os_env.get("V_READ_S25_XOR", "0"), 0))
    v_read_v4_add = ctypes.c_int32(int(os_env.get("V_READ_V4_ADD", "0"), 0))
    v_read_cb = ctypes.c_int32(int(os_env.get("V_READ_CB", "0"), 0))
    v_read_s25_mask = ctypes.c_int32(int(os_env.get("V_READ_S25_MASK", "0"), 0))
    v_read_v2_add = ctypes.c_int32(int(os_env.get("V_READ_V2_ADD", "0"), 0))
    v_read_v3_xor = ctypes.c_int32(int(os_env.get("V_READ_V3_XOR", "0"), 0))
    v_read_v3_add = ctypes.c_int32(int(os_env.get("V_READ_V3_ADD", "0"), 0))
    v_read_base_add = ctypes.c_int32(int(os_env.get("V_READ_BASE_ADD", "0"), 0))
    v_read_s25_override = ctypes.c_int32(int(os_env.get("V_READ_S25_OVERRIDE", "0"), 0))
    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        stride_vh,
        v_read_offset,
        v_read_lane_xor,
        v_read_base_xor,
        v_read_s25_xor,
        v_read_v4_add,
        v_read_cb,
        v_read_s25_mask,
        v_read_v2_add,
        v_read_v3_xor,
        v_read_v3_add,
        v_read_base_add,
        v_read_s25_override,
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,
        256, 1, 1,
        50176,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    raw = out.detach().cpu().view(256, 40)
    bytes_flat = raw[:, :32].contiguous().view(torch.uint8).view(256, 32, 4)
    decoded = decode_fp8_bytes(bytes_flat).reshape(256, 128)
    if v_map in ("colbyte", "blockk"):
        row_idx = bytes_flat.reshape(256, 128).to(torch.int32)
    else:
        row_idx = map_to_codes(decoded, codes) + tile * 32

    packed_bytes = raw[:, 32:40].contiguous().view(torch.uint8).view(256, 8, 4)
    packed_decoded = decode_fp8_bytes(packed_bytes).reshape(256, 32)
    if v_map in ("colbyte", "blockk"):
        packed_row_idx = packed_bytes.reshape(256, 32).to(torch.int32)
    else:
        packed_row_idx = map_to_codes(packed_decoded, codes) + tile * 32

    # Report mapping for first wave (lanes 0..31)
    lane_rows = [int(row_idx[lane, 0].item()) for lane in range(32)]
    print(f"tile={tile} lane->row (pos0) 0..31:", lane_rows)
    for lane in range(8):
        print("lane", lane, "row ids set0:", row_idx[lane, :8].tolist())

    # Show k indices per lane/reg byte for lane 0
    print("lane0 k_idx set0 pos0..15:", row_idx[0, :16].tolist())
    print("lane0 k_idx set1 pos32..47:", row_idx[0, 32:48].tolist())
    print("lane0 k_idx set2 pos64..79:", row_idx[0, 64:80].tolist())
    print("lane0 k_idx set3 pos96..111:", row_idx[0, 96:112].tolist())

    if os_env.get("V_READ_DEBUG", "0") == "1":
        print("lane0 decoded pos28..40:", decoded[0, 28:41].tolist())
        print("lane0 decoded pos48..56:", decoded[0, 48:57].tolist())

    # Dump mapping CSV (raw TR8 read sets)
    out_path = f"{BASE_DIR}/v_read_mapping.csv"
    with open(out_path, "w") as f:
        f.write("lane,pos,k\n")
        for lane in range(64):
            for pos in range(128):
                f.write(f"{lane},{pos},{int(row_idx[lane, pos].item())}\n")
    print(f"wrote {out_path}")

    # Dump packed A-reg mapping (k order)
    packed_path = f"{BASE_DIR}/v_read_packed.csv"
    with open(packed_path, "w") as f:
        f.write("lane,k,row\n")
        for lane in range(64):
            for k in range(32):
                f.write(f"{lane},{k},{int(packed_row_idx[lane, k].item())}\n")
    print(f"wrote {packed_path}")

    print("lane0 packed k->row:", packed_row_idx[0, :16].tolist(), packed_row_idx[0, 16:32].tolist())
    print("lane32 packed k->row:", packed_row_idx[32, :16].tolist(), packed_row_idx[32, 16:32].tolist())


if __name__ == "__main__":
    main()

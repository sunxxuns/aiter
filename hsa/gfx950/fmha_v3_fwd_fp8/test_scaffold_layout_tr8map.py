#!/usr/bin/env python3
"""Apply TR8 address-map swizzle and run scaffold layout test."""
import ctypes
import subprocess
import torch

from test_scaffold_layout import build_identity_qk, run_case  # noqa: E402


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_v_tr8_addr_map_debug.s"
    obj = f"{base_dir}/fwd_fp8_v_tr8_addr_map_debug.o"
    co = f"{base_dir}/fwd_fp8_v_tr8_addr_map_debug.co"
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


def run_pattern(pattern_bytes):
    torch.cuda.init()
    co = build_kernel()

    # Dummy buffers to satisfy scaffold-compatible args
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

    q = torch.zeros((1, 1, s_q, d), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros((1, 1, s_k, d), device="cuda", dtype=torch.float8_e4m3fn)
    v = pattern_bytes.to(device="cuda")

    # Output: 32 dwords per thread (128 bytes)
    out = torch.zeros(256 * 32, device="cuda", dtype=torch.uint32)

    hip, module, func = load_kernel(co, "_fwd_fp8_v_tr8_addr_map_debug")
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

    out_cpu = out.detach().cpu().numpy().astype("uint32")
    hip.hipModuleUnload(module)
    return out_cpu


def words_to_bytes(words):
    out = []
    for w in words:
        out.extend(list(int(w).to_bytes(4, "little")))
    return out


def build_tr8_addr_map():
    size = 8192
    vals = torch.arange(size, dtype=torch.int32)
    low = (vals & 0xFF).to(torch.uint8)
    high = ((vals >> 8) & 0xFF).to(torch.uint8)

    low_words = run_pattern(low)
    high_words = run_pattern(high)

    addr_map = []
    for tid in range(64):
        low_bytes = words_to_bytes(low_words[tid * 32:(tid + 1) * 32])
        high_bytes = words_to_bytes(high_words[tid * 32:(tid + 1) * 32])
        addrs = [lb | (hb << 8) for lb, hb in zip(low_bytes, high_bytes)]
        reads = [addrs[i:i + 8] for i in range(0, len(addrs), 8)]
        addr_map.append(reads)
    return addr_map


def build_swizzle_from_map(addr_map, rows=64, cols=128):
    col_swz = [[None for _ in range(cols)] for _ in range(rows)]
    for tid in range(64):
        for read_idx in range(16):
            block = read_idx // 4
            col_in_block = (tid & 15) + 16 * ((tid >> 5) & 1)
            col = col_in_block + block * 32
            for addr in addr_map[tid][read_idx]:
                row = addr // 128
                col_swz[row][col] = addr % 128
    return col_swz


def apply_swizzle(v, col_swz):
    s = v.shape[-2]
    d = v.shape[-1]
    out = torch.empty_like(v)
    for r in range(s):
        for c in range(d):
            cs = col_swz[r][c]
            if cs is None:
                continue
            out[0, 0, r, cs] = v[0, 0, r, c]
    return out


def main():
    torch.manual_seed(0)
    s = 64
    b, h, d = 1, 1, 128
    num_q_blocks = (s + 127) // 128
    if num_q_blocks % 2 != 0:
        num_q_blocks += 1
    num_k_tiles = (s + 31) // 32
    if num_k_tiles % 2 != 0:
        num_k_tiles += 1
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    qf32, kf32 = build_identity_qk(s_q, s_k, d, device="cuda")
    q = qf32.to(torch.float8_e4m3fn)
    k = kf32.to(torch.float8_e4m3fn)

    v_col = torch.zeros(1, 1, s_k, d, device="cuda", dtype=torch.float32)
    for c in range(d):
        v_col[0, 0, :, c] = c / 128.0

    addr_map = build_tr8_addr_map()
    col_swz = build_swizzle_from_map(addr_map, rows=s_k, cols=d)
    v_swz = apply_swizzle(v_col, col_swz)
    run_case("identity P + V col-pattern (tr8-map)", q, k, v_swz.to(torch.float8_e4m3fn),
             num_q_blocks, num_k_tiles, s_q, s_k, d)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Map TR8 read bytes to (row, col) using actual LDS write swizzle."""
import ctypes
import subprocess
import torch


def build_kernel(src_name):
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/{src_name}"
    obj = src.replace(".s", ".o")
    co = src.replace(".s", ".co")
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


def run_write_addr_debug():
    co = build_kernel("fwd_fp8_v_write_addr_debug.s")
    hip, module, func = load_kernel(co, "_fwd_fp8_v_write_addr_debug")
    out = torch.zeros(256, device="cuda", dtype=torch.uint32)
    q = torch.zeros(1, device="cuda", dtype=torch.uint8)
    k = torch.zeros(1, device="cuda", dtype=torch.uint8)
    v = torch.zeros(1, device="cuda", dtype=torch.uint8)
    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_int32(0),
        ctypes.c_int32(0),
        ctypes.c_int32(0),
        ctypes.c_int32(0),
        ctypes.c_int32(0),
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
    addrs = out.detach().cpu().numpy().astype("uint32").tolist()
    hip.hipModuleUnload(module)
    return addrs


def run_tr8_addr_map():
    co = build_kernel("fwd_fp8_v_tr8_addr_map_debug.s")
    hip, module, func = load_kernel(co, "_fwd_fp8_v_tr8_addr_map_debug")

    # Build 0..8191 byte patterns to recover addresses
    size = 8192
    vals = torch.arange(size, dtype=torch.int32)
    low = (vals & 0xFF).to(torch.uint8)
    high = ((vals >> 8) & 0xFF).to(torch.uint8)

    def run_pattern(pattern_bytes):
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
        out = torch.zeros(256 * 32, device="cuda", dtype=torch.uint32)
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
        return out.detach().cpu().numpy().astype("uint32")

    low_words = run_pattern(low)
    high_words = run_pattern(high)
    hip.hipModuleUnload(module)

    addr_map = []
    for tid in range(64):
        low_bytes = []
        high_bytes = []
        for w in low_words[tid * 32:(tid + 1) * 32]:
            low_bytes.extend(list(int(w).to_bytes(4, "little")))
        for w in high_words[tid * 32:(tid + 1) * 32]:
            high_bytes.extend(list(int(w).to_bytes(4, "little")))
        addrs = [lb | (hb << 8) for lb, hb in zip(low_bytes, high_bytes)]
        reads = [addrs[i:i + 8] for i in range(0, len(addrs), 8)]
        addr_map.append(reads)
    return addr_map


def main():
    write_base = run_write_addr_debug()
    tr8_addrs = run_tr8_addr_map()

    # Build address -> (row, col) map from actual LDS write swizzle
    addr_to_rc = {}
    for tid in range(256):
        row = tid >> 3
        col_block = tid & 7
        base = write_base[tid]
        for i in range(16):
            col = col_block * 16 + i
            addr_to_rc[base + i] = (row, col)
            addr_to_rc[base + 4096 + i] = (row + 32, col)

    # Dump mapping for a few tids
    tids = [0, 1, 2, 3, 8, 9, 10, 11]
    for tid in tids:
        print(f"\n=== tid {tid} ===")
        for ridx, addrs in enumerate(tr8_addrs[tid]):
            rc = [addr_to_rc.get(a, (None, None)) for a in addrs]
            print(f"read {ridx:02d} (row,col): {rc}")


if __name__ == "__main__":
    main()

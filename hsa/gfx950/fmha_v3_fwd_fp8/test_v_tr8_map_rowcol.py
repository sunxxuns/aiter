#!/usr/bin/env python3
"""Map TR8 output bytes to (row, col) using row/col patterns."""
import ctypes
import subprocess
import torch


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


def run_pattern(pattern):
    torch.manual_seed(0)
    torch.cuda.init()

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
    offset = 1
    for r in range(s_k):
        for c in range(d):
            if pattern == "row":
                v_bytes[0, 0, r, c] = (r + offset) & 0xFF
            else:
                v_bytes[0, 0, r, c] = (c + offset) & 0xFF

    q = torch.zeros((1, 1, s_q, d), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros((1, 1, s_k, d), device="cuda", dtype=torch.float8_e4m3fn)

    out = torch.zeros(256 * 32, device="cuda", dtype=torch.uint32)

    co = build_kernel()
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

    out_cpu = out.detach().cpu().numpy().astype("uint32")
    hip.hipModuleUnload(module)
    return out_cpu


def words_to_bytes(words):
    out = []
    for w in words:
        out.extend(list(int(w).to_bytes(4, "little")))
    return out


def main():
    row_words = run_pattern("row")
    col_words = run_pattern("col")
    tids = [0, 1, 2, 3, 8, 9, 10, 11]
    def decode(val):
        return None if val == 0 else int(val) - 1

    for tid in tids:
        row_bytes = words_to_bytes(row_words[tid * 32:(tid + 1) * 32])
        col_bytes = words_to_bytes(col_words[tid * 32:(tid + 1) * 32])
        pairs = [(decode(r), decode(c)) for r, c in zip(row_bytes, col_bytes)]
        chunks = [pairs[i:i + 8] for i in range(0, len(pairs), 8)]
        print(f"\n=== tid {tid} ===")
        for i, chunk in enumerate(chunks):
            print(f"read {i:02d} (row,col):", chunk)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dump TR8 LDS read addresses for a few threads."""
import ctypes
import subprocess
import torch


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_v_tr8_addr_debug.s"
    obj = f"{base_dir}/fwd_fp8_v_tr8_addr_debug.o"
    co = f"{base_dir}/fwd_fp8_v_tr8_addr_debug.co"
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


def main():
    torch.manual_seed(0)
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
    v = torch.zeros((1, 1, s_k, d), device="cuda", dtype=torch.float8_e4m3fn)

    # Output: 16 dwords per thread (64 bytes)
    out = torch.zeros(256 * 16, device="cuda", dtype=torch.uint32)

    hip, module, func = load_kernel(co, "_fwd_fp8_v_tr8_addr_debug")
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
    tids = [0, 1, 2, 3, 8, 9, 10, 11, 32, 33, 34, 35]
    print("addr dump order: v2+0, v2+1088, v2+4096, v2+5184, v3, v4, v5, v6, v2+64, v2+1024, v2+4160, v2+5120, v7, v8, v9, v10")
    for tid in tids:
        addrs = out_cpu[tid * 16:(tid + 1) * 16]
        print(f"\n--- tid {tid} ---")
        print("addrs:", addrs.tolist())
        rows = [(int(a) // 128, int(a) % 128) for a in addrs]
        print("row,col:", rows)

    hip.hipModuleUnload(module)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dump bitop3:0x78 LDS write addresses."""
import ctypes
import subprocess
import torch


def build_kernel():
    base_dir = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
    src = f"{base_dir}/fwd_fp8_v_write_addr_debug.s"
    obj = f"{base_dir}/fwd_fp8_v_write_addr_debug.o"
    co = f"{base_dir}/fwd_fp8_v_write_addr_debug.co"
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
    torch.cuda.init()
    co = build_kernel()

    # Dummy buffers to satisfy scaffold-compatible args
    out = torch.zeros(256, device="cuda", dtype=torch.uint32)
    q = torch.zeros(1, device="cuda", dtype=torch.uint8)
    k = torch.zeros(1, device="cuda", dtype=torch.uint8)
    v = torch.zeros(1, device="cuda", dtype=torch.uint8)

    hip, module, func = load_kernel(co, "_fwd_fp8_v_write_addr_debug")
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

    out_cpu = out.detach().cpu().numpy().astype("uint32")
    uniq = len(set(int(x) for x in out_cpu))
    print("unique addrs:", uniq)
    for tid in range(16):
        print(f"tid {tid} addr {int(out_cpu[tid])}")

    hip.hipModuleUnload(module)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify MFMA K=64 mapping hypothesis via direct VGPR inputs."""
import ctypes
import os
import subprocess
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "0"


BASE_DIR = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8"
SRC = f"{BASE_DIR}/fwd_fp8_mfma_map_debug.s"
CO = f"{BASE_DIR}/fwd_fp8_mfma_map_debug.co"


def build_kernel():
    obj = f"{BASE_DIR}/fwd_fp8_mfma_map_debug.o"
    cmd = (
        f"/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa "
        f"-mcpu=gfx950 -mno-xnack -c {SRC} -o {obj} && "
        f"/opt/rocm/llvm/bin/ld.lld -shared {obj} -o {CO}"
    )
    subprocess.check_call(cmd, shell=True)
    return CO


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


def decode_fp8_bytes(byte_tensor):
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return table[byte_tensor.long()]


def build_codebook(count=64):
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    finite = table[torch.isfinite(table)]
    finite = finite[finite.abs() >= 0.5]
    unique = torch.unique(finite)
    unique, _ = torch.sort(unique)
    if unique.numel() < count:
        raise RuntimeError(f"Not enough FP8 codes: {unique.numel()}")
    idx = torch.linspace(0, unique.numel() - 1, steps=count).round().long()
    codes = unique[idx]
    # Map values back to byte codes
    codebook = []
    for val in codes:
        mask = table == val
        byte = torch.nonzero(mask, as_tuple=False)[0].item()
        codebook.append(byte)
    return torch.tensor(codebook, dtype=torch.uint8)


def mapping_hypothesis(lane, reg, byte):
    """Return (m_or_n, k) for A/B mapping hypothesis."""
    # Thread-to-N mapping per ISA hypothesis
    mn = (lane % 16) + 16 * ((lane // 32) % 2)
    # k ranges by lane group and reg index
    lane_group = lane // 16  # 0..3
    if lane_group in (0, 1):
        k_base0 = 0
        k_base1 = 32
    else:
        k_base0 = 16
        k_base1 = 48
    if reg < 4:
        k = k_base0 + reg * 4 + byte
    else:
        k = k_base1 + (reg - 4) * 4 + byte
    return mn, k


def build_inputs(codebook):
    """Build packed A/B regs per lane with deterministic byte codes."""
    lanes = 64
    regs = 16  # 8 for A, 8 for B
    packed = torch.zeros(lanes, regs, dtype=torch.uint32)

    map_a = os.environ.get("MFMA_MAP_A", "0") == "1"
    if map_a:
        # A: 32 unique byte codes per reg/byte, identical for all lanes.
        # B: identity selector for k in [0..31] (k==n) via mapping_hypothesis.
        codebook32 = codebook[:32]
        for lane in range(lanes):
            lane_group = lane // 16
            if lane_group in (0, 2):
                k_base0, k_base1 = 16, 0
            else:
                k_base0, k_base1 = 48, 32
            for reg in range(8):  # A regs
                bytes_out = []
                for b in range(4):
                    code_idx = reg * 4 + b
                    code = int(codebook32[code_idx].item())
                    bytes_out.append(code)
                word = bytes_out[0] | (bytes_out[1] << 8) | (bytes_out[2] << 16) | (bytes_out[3] << 24)
                packed[lane, reg] = word
            for reg in range(8):  # B regs (selector)
                bytes_out = []
                for b in range(4):
                    n = (lane % 16) + 16 * ((lane // 32) % 2)
                    if reg < 4:
                        k = k_base0 + reg * 4 + b
                    else:
                        k = k_base1 + (reg - 4) * 4 + b
                    if 0 <= k < 32 and k == n:
                        code = 0x38  # FP8 1.0
                    else:
                        code = 0x00  # FP8 0.0
                    bytes_out.append(code)
                word = bytes_out[0] | (bytes_out[1] << 8) | (bytes_out[2] << 16) | (bytes_out[3] << 24)
                packed[lane, 8 + reg] = word
        return packed

    for lane in range(lanes):
        for reg in range(8):  # A regs
            bytes_out = []
            for b in range(4):
                code_idx = (lane * 32 + reg * 4 + b) % 64
                code = int(codebook[code_idx].item())
                bytes_out.append(code)
            word = bytes_out[0] | (bytes_out[1] << 8) | (bytes_out[2] << 16) | (bytes_out[3] << 24)
            packed[lane, reg] = word
        for reg in range(8):  # B regs
            bytes_out = []
            for b in range(4):
                code_idx = (lane * 32 + (reg + 8) * 4 + b) % 64
                code = int(codebook[code_idx].item())
                bytes_out.append(code)
            word = bytes_out[0] | (bytes_out[1] << 8) | (bytes_out[2] << 16) | (bytes_out[3] << 24)
            packed[lane, 8 + reg] = word
    return packed


def decode_inputs(packed):
    """Decode packed bytes to float values by lane/reg/byte."""
    bytes_flat = packed.view(torch.uint8).view(64, 16, 4)
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    decoded = table[bytes_flat.long()]
    return decoded


def decode_output(raw):
    """Decode v32-v47 output to row-major 32x32."""
    decoded = torch.empty((32, 32), dtype=torch.float32)
    raw_mat = raw.view(64, 16)
    for tid in range(64):
        lane = tid & 63
        col = lane & 31
        row_base = ((lane >> 5) & 1) * 4
        for i in range(16):
            row = (i % 4) + row_base + (i // 4) * 8
            decoded[row, col] = raw_mat[tid, i]
    return decoded


def build_ab(decoded, mapping):
    """Build A (32x64) and B (64x32) using mapping hypothesis."""
    A = torch.zeros(32, 64, dtype=torch.float32)
    B = torch.zeros(64, 32, dtype=torch.float32)
    for lane in range(64):
        m = (lane % 16) + 16 * ((lane // 32) % 2)
        n = m
        group = lane // 16
        low_base, high_base = mapping[group]
        # A regs
        for reg in range(8):
            base = low_base if reg < 4 else high_base
            for b in range(4):
                k = base + (reg % 4) * 4 + b
                A[m, k] = decoded[lane, reg, b].item()
        # B regs
        for reg in range(8):
            base = low_base if reg < 4 else high_base
            for b in range(4):
                k = base + (reg % 4) * 4 + b
                B[k, n] = decoded[lane, 8 + reg, b].item()
    return A, B


def main():
    torch.manual_seed(0)
    codebook = build_codebook(64)
    packed = build_inputs(codebook)

    co = build_kernel()
    hip, module, func = load_kernel(co, "_fwd_fp8_mfma_map_debug")

    in_buf = packed.to(device="cuda")
    out = torch.zeros(64 * 16, device="cuda", dtype=torch.float32)

    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(in_buf.data_ptr()),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )
    hip.hipModuleLaunchKernel(
        func,
        1, 1, 1,
        64, 1, 1,
        0,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)

    raw = out.detach().cpu()
    decoded_out = decode_output(raw)
    decoded_in = decode_inputs(packed)

    if os.environ.get("MFMA_MAP_A", "0") == "1":
        # Map output column n -> A reg/byte via codebook index.
        codebook32 = codebook[:32]
        table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        value_to_idx = {}
        for idx, code in enumerate(codebook32):
            code_val = int(code.item())
            value_to_idx[float(table[code_val].item())] = idx
        mapping = {}
        for n in range(32):
            val = float(decoded_out[0, n].item())
            idx = value_to_idx.get(val, -1)
            if idx >= 0:
                reg = idx // 4
                byte = idx % 4
                mapping[n] = (reg, byte)
        print("A map (k=n -> reg,byte):", mapping)
        print("decoded_out[0:8,0:8]:")
        print(decoded_out[:8, :8])
        return

    # Search mapping hypotheses (all groups independent)
    bases = [0, 16, 32, 48]
    best = None
    for low01 in bases:
        for high01 in bases:
            if high01 == low01:
                continue
            for low1 in bases:
                for high1 in bases:
                    if high1 == low1:
                        continue
                    for low2 in bases:
                        for high2 in bases:
                            if high2 == low2:
                                continue
                            for low3 in bases:
                                for high3 in bases:
                                    if high3 == low3:
                                        continue
                                    mapping = {
                                        0: (low01, high01),
                                        1: (low1, high1),
                                        2: (low2, high2),
                                        3: (low3, high3),
                                    }
                                    A, B = build_ab(decoded_in, mapping)
                                    ref = A @ B
                                    # allow global scale mismatch
                                    scale = (decoded_out * ref).sum() / (ref * ref).sum()
                                    ref_scaled = ref * scale
                                    diff = (decoded_out - ref_scaled).abs()
                                    mean_err = diff.mean().item()
                                    if best is None or mean_err < best[0]:
                                        best = (mean_err, mapping, diff.max().item(), scale.item())

    print("=== MFMA map debug ===")
    print("best mean_err:", best[0], "max_err:", best[2], "scale:", best[3], "mapping:", best[1])
    print("decoded[0:8,0:8]:")
    print(decoded_out[:8, :8])
    A, B = build_ab(decoded_in, best[1])
    ref = A @ B * best[3]
    print("ref[0:8,0:8]:")
    print(ref[:8, :8])


if __name__ == "__main__":
    main()

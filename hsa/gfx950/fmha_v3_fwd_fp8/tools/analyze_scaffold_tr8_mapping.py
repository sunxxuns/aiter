#!/usr/bin/env python3
"""
Analyze scaffold PV TR8 mapping deterministically (no statistics).

We run 4 dump-and-exit passes to disambiguate where packed A bytes come from:
  - raw TR8 regs dump (v200..v231) with V pattern P0
  - raw TR8 regs dump (v200..v231) with V pattern P1
  - packed A dump (v48..v55)      with V pattern P0
  - packed A dump (v48..v55)      with V pattern P1

For each thread tid, each raw byte position r has a signature (P0_val, P1_val),
and each packed byte position p has a signature (P0_val, P1_val).

When a signature is unique among raw bytes, we can map packed[p] -> raw[r].

This gives a concrete, per-lane mapping target to fix the pack logic.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _load_hip() -> ctypes.CDLL:
    return ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")


def _unpack_u32_bytes(u: int) -> List[int]:
    return [(u >> (8 * i)) & 0xFF for i in range(4)]


def _make_V(*, s_k: int, D: int, pattern: str) -> torch.Tensor:
    vb = torch.empty((s_k, D), device="cuda", dtype=torch.uint8)
    if pattern == "rowbyte":
        for k in range(s_k):
            vb[k, :] = k & 0xFF
    elif pattern == "rowxorcol":
        cols = torch.arange(D, device="cuda", dtype=torch.uint8)
        for k in range(s_k):
            vb[k, :] = (cols ^ (k & 0xFF)) & 0xFF
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return vb.view(torch.float8_e4m3fn).reshape(1, 1, s_k, D)


def _launch(
    hip: ctypes.CDLL,
    *,
    out: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_k_tiles: int,
    stride_qh: int,
    stride_kh: int,
    stride_vh: int,
    stride_oh_bytes: int,
    debug_flags: int,
    v_read_cb: int,
    v_read_lane_add: int,
    v_read_v3_xor: int,
    v_read_v3_add: int,
    v_read_v4_add: int,
    v_read_v2_add: int,
    v_read_base_add: int,
    v_read_base_xor: int,
    v_read_base_extra_add: int,
    v_read_s25_override: int,
) -> None:
    args = [
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_int32(num_k_tiles),
        ctypes.c_int32(stride_qh),
        ctypes.c_int32(stride_kh),
        ctypes.c_int32(stride_vh),
        ctypes.c_int32(stride_oh_bytes),
        ctypes.c_int32(debug_flags),
        ctypes.c_int32(v_read_cb),
        ctypes.c_int32(v_read_lane_add),
        ctypes.c_int32(v_read_v3_xor),
        ctypes.c_int32(v_read_v3_add),
        ctypes.c_int32(v_read_v4_add),
        ctypes.c_int32(v_read_v2_add),
        ctypes.c_int32(v_read_base_add),
        ctypes.c_int32(v_read_base_xor),
        ctypes.c_int32(v_read_base_extra_add),
        ctypes.c_int32(v_read_s25_override),
    ]
    args_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    hip.hipModuleLoad(ctypes.byref(module), CO_PATH.encode())
    hip.hipModuleGetFunction(ctypes.byref(func), module, KERNEL_NAME)
    hip.hipModuleLaunchKernel(func, 1, 1, 1, 512, 1, 1, 50176, None, args_ptrs, None)
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)


def _dump_raw(
    hip: ctypes.CDLL,
    *,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_k_tiles: int,
    stride_qh: int,
    stride_kh: int,
    stride_vh: int,
    v_read_cb: int,
    lane_add: int,
    v3_xor: int,
    v3_add: int,
    v4_add: int,
    v2_add: int,
    base_add: int,
    base_xor: int,
    base_extra: int,
    s25: int,
) -> List[int]:
    out = torch.zeros(512 * 32, device="cuda", dtype=torch.uint32)  # 128B/tid
    flags = 0x01000000 | 0x00000004 | 0x00000080
    if os.environ.get("MAP_IDENTITY_WRITE", "1") == "1":
        flags |= 0x00080000
    _launch(
        hip,
        out=out,
        Q=Q,
        K=K,
        V=V,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        stride_oh_bytes=out.numel() * 4,
        debug_flags=flags,
        v_read_cb=v_read_cb,
        v_read_lane_add=lane_add,
        v_read_v3_xor=v3_xor,
        v_read_v3_add=v3_add,
        v_read_v4_add=v4_add,
        v_read_v2_add=v2_add,
        v_read_base_add=base_add,
        v_read_base_xor=base_xor,
        v_read_base_extra_add=base_extra,
        v_read_s25_override=s25,
    )
    return out.cpu().tolist()


def _dump_packed(
    hip: ctypes.CDLL,
    *,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_k_tiles: int,
    stride_qh: int,
    stride_kh: int,
    stride_vh: int,
    v_read_cb: int,
    lane_add: int,
    v3_xor: int,
    v3_add: int,
    v4_add: int,
    v2_add: int,
    base_add: int,
    base_xor: int,
    base_extra: int,
    s25: int,
) -> List[int]:
    out = torch.zeros(512 * 8, device="cuda", dtype=torch.uint32)  # 32B/tid
    flags = 0x00200000 | 0x00000004 | 0x00000080
    if os.environ.get("MAP_IDENTITY_WRITE", "1") == "1":
        flags |= 0x00080000
    _launch(
        hip,
        out=out,
        Q=Q,
        K=K,
        V=V,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        stride_oh_bytes=out.numel() * 4,
        debug_flags=flags,
        v_read_cb=v_read_cb,
        v_read_lane_add=lane_add,
        v_read_v3_xor=v3_xor,
        v_read_v3_add=v3_add,
        v_read_v4_add=v4_add,
        v_read_v2_add=v2_add,
        v_read_base_add=base_add,
        v_read_base_xor=base_xor,
        v_read_base_extra_add=base_extra,
        v_read_s25_override=s25,
    )
    return out.cpu().tolist()


@dataclass(frozen=True)
class MappingStats:
    mapped_bytes: int
    ambiguous_bytes: int
    unmapped_bytes: int


def main() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")

    B, H, D = 1, 1, 128
    num_q_blocks = 2
    num_k_tiles = 2
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    Q = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float8_e4m3fn)
    K = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)
    V0 = _make_V(s_k=s_k, D=D, pattern="rowbyte")
    V1 = _make_V(s_k=s_k, D=D, pattern="rowxorcol")

    colblk = int(os.environ.get("MAP_COLBLK", "0")) & 3
    v23_tweak = int(os.environ.get("MAP_V23_TWEAK", "0")) & 0xF
    v24_tweak = int(os.environ.get("MAP_V24_TWEAK", "0")) & 0xF
    v24_delta_idx = int(os.environ.get("MAP_V24_DELTA_IDX", "0")) & 0xFF
    perm_id = int(os.environ.get("MAP_PERM_ID", "0"))
    write_mode = int(os.environ.get("MAP_WRITE_MODE", "2"))
    v_read_cb = (
        colblk
        | (perm_id << 2)
        | (write_mode << 8)
        | (v23_tweak << 12)
        | (v24_tweak << 16)
        | (v24_delta_idx << 20)
    )
    lane_add = int(os.environ.get("MAP_LANE_ADD", "0"))
    v3_xor = int(os.environ.get("MAP_V3_XOR", "0"))
    v3_add = int(os.environ.get("MAP_V3_ADD", "0"))
    v4_add = int(os.environ.get("MAP_V4_ADD", "0"))
    v2_add = int(os.environ.get("MAP_V2_ADD", "0"))
    base_add = int(os.environ.get("MAP_BASE_ADD", "0"), 0)
    base_xor = int(os.environ.get("MAP_BASE_XOR", "0"), 0)
    base_extra = int(os.environ.get("MAP_BASE_EXTRA_ADD", "0"), 0)
    s25 = int(os.environ.get("MAP_S25", "0"), 0)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D

    raw0 = _dump_raw(
        hip,
        Q=Q,
        K=K,
        V=V0,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        v_read_cb=v_read_cb,
        lane_add=lane_add,
        v3_xor=v3_xor,
        v3_add=v3_add,
        v4_add=v4_add,
        v2_add=v2_add,
        base_add=base_add,
        base_xor=base_xor,
        base_extra=base_extra,
        s25=s25,
    )
    raw1 = _dump_raw(
        hip,
        Q=Q,
        K=K,
        V=V1,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        v_read_cb=v_read_cb,
        lane_add=lane_add,
        v3_xor=v3_xor,
        v3_add=v3_add,
        v4_add=v4_add,
        v2_add=v2_add,
        base_add=base_add,
        base_xor=base_xor,
        base_extra=base_extra,
        s25=s25,
    )
    a0 = _dump_packed(
        hip,
        Q=Q,
        K=K,
        V=V0,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        v_read_cb=v_read_cb,
        lane_add=lane_add,
        v3_xor=v3_xor,
        v3_add=v3_add,
        v4_add=v4_add,
        v2_add=v2_add,
        base_add=base_add,
        base_xor=base_xor,
        base_extra=base_extra,
        s25=s25,
    )
    a1 = _dump_packed(
        hip,
        Q=Q,
        K=K,
        V=V1,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        v_read_cb=v_read_cb,
        lane_add=lane_add,
        v3_xor=v3_xor,
        v3_add=v3_add,
        v4_add=v4_add,
        v2_add=v2_add,
        base_add=base_add,
        base_xor=base_xor,
        base_extra=base_extra,
        s25=s25,
    )

    def raw_sig_for_tid(tid: int) -> Dict[Tuple[int, int], List[int]]:
        # returns sig->list(raw_byte_pos)
        sig2pos: Dict[Tuple[int, int], List[int]] = {}
        base = tid * 32
        # 32 dwords -> 128 bytes
        bpos = 0
        for j in range(32):
            u0 = int(raw0[base + j])
            u1 = int(raw1[base + j])
            b0 = _unpack_u32_bytes(u0)
            b1 = _unpack_u32_bytes(u1)
            for k in range(4):
                sig = (b0[k], b1[k])
                sig2pos.setdefault(sig, []).append(bpos)
                bpos += 1
        return sig2pos

    def packed_sig_for_tid(tid: int) -> List[Tuple[int, int]]:
        base = tid * 8
        sigs: List[Tuple[int, int]] = []
        for j in range(8):
            u0 = int(a0[base + j])
            u1 = int(a1[base + j])
            b0 = _unpack_u32_bytes(u0)
            b1 = _unpack_u32_bytes(u1)
            for k in range(4):
                sigs.append((b0[k], b1[k]))
        return sigs  # len 32

    def raw_rowbyte_values_for_tid(tid: int) -> List[int]:
        base = tid * 32
        bs: List[int] = []
        for j in range(32):
            u0 = int(raw0[base + j])
            bs += _unpack_u32_bytes(u0)
        return bs  # 128 bytes

    # Analyze mapping for tid0 + tid32, plus summary stats across all tids.
    def analyze_tid(tid: int) -> MappingStats:
        sig2pos = raw_sig_for_tid(tid)
        psigs = packed_sig_for_tid(tid)
        mapped = 0
        amb = 0
        unm = 0
        for sig in psigs:
            poss = sig2pos.get(sig)
            if not poss:
                unm += 1
            elif len(poss) == 1:
                mapped += 1
            else:
                amb += 1
        return MappingStats(mapped_bytes=mapped, ambiguous_bytes=amb, unmapped_bytes=unm)

    for tid in (0, 32):
        st = analyze_tid(tid)
        lane = tid & 63
        print(f"tid={tid} lane={lane} mapped={st.mapped_bytes}/32 ambiguous={st.ambiguous_bytes} unmapped={st.unmapped_bytes}")

    # Global worst-case mapped bytes
    min_mapped = 32
    min_tid = -1
    total_mapped = 0
    for tid in range(512):
        st = analyze_tid(tid)
        total_mapped += st.mapped_bytes
        if st.mapped_bytes < min_mapped:
            min_mapped = st.mapped_bytes
            min_tid = tid
    print(f"\nGLOBAL: min_mapped={min_mapped}/32 at tid={min_tid}; avg_mapped={total_mapped/(512*32):.3f}")

    # Also report raw TR8 value coverage (rowbyte) deterministically.
    min_cov = 32
    worst_cov_tid = -1
    worst_missing: List[int] = []
    max_bad = 0
    worst_bad_tid = -1
    for tid in range(512):
        lane = tid & 63
        expect = set(range(0, 32)) if lane < 32 else set(range(32, 64))
        bs = raw_rowbyte_values_for_tid(tid)
        present = set(b for b in bs if b <= 63)
        cov = len(present & expect)
        bad = sum(1 for b in bs if b > 63)
        if cov < min_cov:
            min_cov = cov
            worst_cov_tid = tid
            missing = sorted(expect - present)
            worst_missing = missing
        if bad > max_bad:
            max_bad = bad
            worst_bad_tid = tid
    print(
        f"RAW COVERAGE: min_cov={min_cov}/32 at tid={worst_cov_tid} missing={worst_missing} ; "
        f"max_bad={max_bad}/128 at tid={worst_bad_tid}"
    )


if __name__ == "__main__":
    main()


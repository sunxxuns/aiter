#!/usr/bin/env python3
"""
Dump scaffold PV TR8 reads and packed MFMA-A bytes deterministically.

This runs the scaffold kernel twice with the same knobs:
  1) debug_flags=0x01000000: dumps raw TR8 regs v200..v231 (128B/tid)
  2) debug_flags=0x00000001: dumps PV operands right before MFMA and exits:
       - B operand (packed V) in v0..v7   (32B/tid)
       - A operand (packed P) in v48..v55 (32B/tid)

Both use:
  - debug_flags |= 0x00000004  (use v_read_cb-style PV base/knobs)
  - debug_flags |= 0x00000080  (disable V row permutation)

Optionally:
  - debug_flags |= 0x00080000  (force identity V-write layout) if DUMP_IDENTITY_WRITE=1

We initialize V as rowbyte (V[k,*]=k) so that the ISA expected MFMA input layout
is easy to check (CDNA4 ISA 7.1.5.1, FP8 32x32x64):
  - lanes 0..31 should pack bytes [0..15, 32..47] (in order)
  - lanes 32..63 should pack bytes [16..31, 48..63] (in order)

This script prints:
  - lane0 / lane32 raw-byte coverage (set) from TR8 regs
  - lane0 / lane32 packed bytes and positional-match counts
  - global worst-case positional match across all 512 threads
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Tuple

import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _unpack_u32_bytes(u: int) -> List[int]:
    return [(u >> (8 * i)) & 0xFF for i in range(4)]


def _score_positional(got32: List[int], exp32: List[int]) -> int:
    return sum(1 for i in range(32) if got32[i] == (exp32[i] & 0xFF))


def _load_hip() -> ctypes.CDLL:
    return ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")


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


def main() -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")

    # Fixed one-block shape
    B, H, D = 1, 1, 128
    num_q_blocks = int(os.environ.get("DUMP_Q_BLOCKS", "2"))
    num_k_tiles = int(os.environ.get("DUMP_K_TILES", "2"))
    s_q = num_q_blocks * 128
    s_k = num_k_tiles * 32

    # Default to identity-P (Q/K identity) so the repro matches numerics identity-P semantics.
    qk_identity = os.environ.get("DUMP_IDENTITY_P", "1") == "1"
    if qk_identity:
        Qf32 = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float32)
        Kf32 = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float32)
        for r in range(min(s_k, D, s_q)):
            Qf32[0, 0, r, r] = 1.0
            Kf32[0, 0, r, r] = 1.0
        Q = Qf32.to(torch.float8_e4m3fn)
        K = Kf32.to(torch.float8_e4m3fn)
    else:
        Q = torch.zeros((B, H, s_q, D), device="cuda", dtype=torch.float8_e4m3fn)
        K = torch.zeros((B, H, s_k, D), device="cuda", dtype=torch.float8_e4m3fn)

    v_bytes = torch.empty((s_k, D), device="cuda", dtype=torch.uint8)
    for r in range(s_k):
        v_bytes[r, :] = r & 0xFF
    V = v_bytes.view(torch.float8_e4m3fn).reshape(B, H, s_k, D)

    # Knobs (env override friendly)
    colblk = int(os.environ.get("DUMP_COLBLK", "0")) & 3
    v23_tweak = int(os.environ.get("DUMP_V23_TWEAK", "0")) & 0xF
    v24_tweak = int(os.environ.get("DUMP_V24_TWEAK", "0")) & 0xF
    v24_delta_idx = int(os.environ.get("DUMP_V24_DELTA_IDX", "0")) & 0xFF
    perm_id = int(os.environ.get("DUMP_PERM_ID", "0"))
    write_mode = int(os.environ.get("DUMP_WRITE_MODE", "2"))
    v_read_cb = (
        colblk
        | (perm_id << 2)
        | (write_mode << 8)
        | (v23_tweak << 12)
        | (v24_tweak << 16)
        | (v24_delta_idx << 20)
    )
    v_read_lane_add = int(os.environ.get("DUMP_LANE_ADD", "0"))
    v_read_v3_xor = int(os.environ.get("DUMP_V3_XOR", "0"))
    v_read_v3_add = int(os.environ.get("DUMP_V3_ADD", "0"))
    v_read_v4_add = int(os.environ.get("DUMP_V4_ADD", "0"))
    v_read_v2_add = int(os.environ.get("DUMP_V2_ADD", "0"))
    v_read_base_add = int(os.environ.get("DUMP_BASE_ADD", "0"), 0)
    v_read_base_xor = int(os.environ.get("DUMP_BASE_XOR", "0"), 0)
    v_read_base_extra_add = int(os.environ.get("DUMP_BASE_EXTRA_ADD", "0"), 0)
    v_read_s25_override = int(os.environ.get("DUMP_S25", "0"), 0)

    stride_qh = s_q * D
    stride_kh = s_k * D
    stride_vh = s_k * D

    disable_row_permute = os.environ.get("DUMP_DISABLE_ROW_PERMUTE", "1") == "1"
    base_flags = 0x00000004
    if disable_row_permute:
        base_flags |= 0x00000080

    # Optional: only dump PV AB (skip raw TR8 dump) for fast iteration.
    only_pv_ab = os.environ.get("DUMP_ONLY_PV_AB", "0") == "1"
    raw_u32: List[int] = []
    if not only_pv_ab:
        # --- raw TR8 regs dump (v200..v231): 32 dwords per tid
        out_raw = torch.zeros(512 * 32, device="cuda", dtype=torch.uint32)
        flags_raw = 0x01000000 | base_flags
        if os.environ.get("DUMP_IDENTITY_WRITE", "1") == "1":
            flags_raw |= 0x00080000
        _launch(
            hip,
            out=out_raw,
            Q=Q,
            K=K,
            V=V,
            num_k_tiles=num_k_tiles,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_oh_bytes=out_raw.numel() * 4,
            debug_flags=flags_raw,
            v_read_cb=v_read_cb,
            v_read_lane_add=v_read_lane_add,
            v_read_v3_xor=v_read_v3_xor,
            v_read_v3_add=v_read_v3_add,
            v_read_v4_add=v_read_v4_add,
            v_read_v2_add=v_read_v2_add,
            v_read_base_add=v_read_base_add,
            v_read_base_xor=v_read_base_xor,
            v_read_base_extra_add=v_read_base_extra_add,
            v_read_s25_override=v_read_s25_override,
        )
        raw_u32 = out_raw.cpu().tolist()

    # --- PV operand dump (B=v0..v7, A=v48..v55): 16 dwords per tid
    out_ab = torch.zeros(512 * 16, device="cuda", dtype=torch.uint32)
    flags_ab = 0x00000001 | base_flags
    if os.environ.get("DUMP_IDENTITY_WRITE", "1") == "1":
        flags_ab |= 0x00080000
    _launch(
        hip,
        out=out_ab,
        Q=Q,
        K=K,
        V=V,
        num_k_tiles=num_k_tiles,
        stride_qh=stride_qh,
        stride_kh=stride_kh,
        stride_vh=stride_vh,
        stride_oh_bytes=out_ab.numel() * 4,
        debug_flags=flags_ab,
        v_read_cb=v_read_cb,
        v_read_lane_add=v_read_lane_add,
        v_read_v3_xor=v_read_v3_xor,
        v_read_v3_add=v_read_v3_add,
        v_read_v4_add=v_read_v4_add,
        v_read_v2_add=v_read_v2_add,
        v_read_base_add=v_read_base_add,
        v_read_base_xor=v_read_base_xor,
        v_read_base_extra_add=v_read_base_extra_add,
        v_read_s25_override=v_read_s25_override,
    )

    ab_u32 = out_ab.cpu().tolist()

    def raw_bytes_for_tid(tid: int) -> List[int]:
        if only_pv_ab:
            return [0] * 128
        base = tid * 32
        bs: List[int] = []
        for u in raw_u32[base : base + 32]:
            bs += _unpack_u32_bytes(int(u))
        return bs

    def b_bytes_for_tid(tid: int) -> List[int]:
        # v0..v7 (8 dwords)
        base = tid * 16
        bs: List[int] = []
        for u in ab_u32[base : base + 8]:
            bs += _unpack_u32_bytes(int(u))
        return bs

    def a_bytes_for_tid(tid: int) -> List[int]:
        # v48..v55 (8 dwords), stored after v0..v7
        base = tid * 16 + 8
        bs: List[int] = []
        for u in ab_u32[base : base + 8]:
            bs += _unpack_u32_bytes(int(u))
        return bs

    exp0 = list(range(0, 16)) + list(range(32, 48))
    exp32 = list(range(16, 32)) + list(range(48, 64))

    for tid in (0, 32):
        lane = tid & 63
        exp = exp0 if lane < 32 else exp32
        rbs = raw_bytes_for_tid(tid)
        bbs = b_bytes_for_tid(tid)
        abs_ = a_bytes_for_tid(tid)
        print(f"tid={tid} lane={lane}")
        print(f"  raw unique={len(set(rbs))} contains_expected={len(set(rbs) & set(exp))}/32")
        print(f"  packed V(B) bytes: {bbs[:32]}")
        print(f"  packed V(B) positional: {_score_positional(bbs, exp)}/32")
        if qk_identity:
            print(f"  packed P(A) bytes: {abs_[:32]}")

    # Global worst-case across all tids
    min_pos = 32
    perfect = 0
    for tid in range(512):
        lane = tid & 63
        exp = exp0 if lane < 32 else exp32
        bbs = b_bytes_for_tid(tid)
        pos = _score_positional(bbs, exp)
        min_pos = min(min_pos, pos)
        if pos == 32:
            perfect += 1
    print(f"\nGLOBAL: min_pos={min_pos} perfect_threads={perfect}/512")


if __name__ == "__main__":
    main()


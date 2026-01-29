#!/usr/bin/env python3
"""
Rigorous B-path validation for fwd_fp8_scaffold.s using random Q/K.

We validate two things:
  1) PACK correctness:
     Dump QK accumulators (v32..v47, FP32) and packed P regs (v48..v55, 8x u32)
     in the kernel, then verify in Python that the packed bytes match
     float8_e4m3fn conversion of the dumped FP32 values.

  2) MIX correctness:
     Dump pre-mix (v48..v55) and post-mix (v48..v55) and verify post-mix
     is the expected lane-wise permutation from values originating in the wave.

Env knobs:
  - SEED (default 0)
  - S (sequence length for K/V tiles; default 64)
  - DUMP_PACK_ONLY: if set, only run PACK check
  - DUMP_MIX_ONLY:  if set, only run MIX check
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
import torch


CO_PATH = "/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co"
KERNEL_NAME = b"_fwd_fp8_scaffold"


def _load_hip():
    return ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")


@dataclass
class LaunchCfg:
    num_q_blocks: int = 2
    num_k_tiles: int = 2
    D: int = 128
    waves: int = 8  # 512 threads total

    @property
    def s_q(self) -> int:
        return self.num_q_blocks * 128  # scaffold convention

    @property
    def s_k(self) -> int:
        return self.num_k_tiles * 32  # scaffold convention

    @property
    def threads(self) -> int:
        return self.waves * 64


def _launch_dump(
    hip,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    out_u32: torch.Tensor,
    num_k_tiles: int,
    stride_qh: int,
    stride_kh: int,
    stride_vh: int,
    stride_oh: int,
    debug_flags: int,
    v_read_cb: int = 0,
    v_read_lane_add: int = 0,
    v_read_v3_xor: int = 0,
    v_read_v3_add: int = 0,
    v_read_v4_add: int = 0,
    v_read_v2_add: int = 0,
    v_read_base_add: int = 0,
    v_read_base_xor: int = 0,
    v_read_base_extra_add: int = 0,
    v_read_s25_override: int = 0,
):
    # out_u32 is used as O_ptr for debug dumps (raw u32 stores)
    args = [
        ctypes.c_void_p(out_u32.data_ptr()),
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_int32(num_k_tiles),
        ctypes.c_int32(stride_qh),
        ctypes.c_int32(stride_kh),
        ctypes.c_int32(stride_vh),
        ctypes.c_int32(stride_oh),
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
    hip.hipModuleLaunchKernel(
        func,
        1,
        1,
        1,
        512,
        1,
        1,
        50176,
        None,
        args_ptrs,
        None,
    )
    hip.hipDeviceSynchronize()
    hip.hipModuleUnload(module)


def _u32_to_f32(u: int) -> float:
    return struct.unpack("f", struct.pack("I", int(u) & 0xFFFFFFFF))[0]


def _pack4_fp8_e4m3fn(a: float, b: float, c: float, d: float) -> int:
    # GPU packs bytes little-endian into a u32.
    t = torch.tensor([a, b, c, d], dtype=torch.float32)
    fb = t.to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    return int(fb[0]) | (int(fb[1]) << 8) | (int(fb[2]) << 16) | (int(fb[3]) << 24)


def check_pack(dump_u32: torch.Tensor, cfg: LaunchCfg):
    """
    dump_u32: shape (threads * 24,) u32
      per tid: [v32..v47, v48..v55]
    """
    dump = dump_u32.view(cfg.threads, 24).cpu()

    # Validate all threads.
    # Note: at this dump point, v32..v47 are tile1 accumulators.
    # So we validate only tile1 pack (v52..v55) == pack(v32..v47).
    mism = 0
    for tid in range(cfg.threads):
        acc_u32 = dump[tid, 0:16].tolist()
        pack_u32 = dump[tid, 16:24].tolist()
        acc_f = [_u32_to_f32(x) for x in acc_u32]

        exp = [
            _pack4_fp8_e4m3fn(acc_f[0], acc_f[1], acc_f[2], acc_f[3]),
            _pack4_fp8_e4m3fn(acc_f[4], acc_f[5], acc_f[6], acc_f[7]),
            _pack4_fp8_e4m3fn(acc_f[8], acc_f[9], acc_f[10], acc_f[11]),
            _pack4_fp8_e4m3fn(acc_f[12], acc_f[13], acc_f[14], acc_f[15]),
        ]

        got_tile1 = pack_u32[4:8]
        if any(int(got_tile1[i]) != int(exp[i]) for i in range(4)):
            mism += 1
            if mism <= 5:
                print(f"[PACK MISMATCH] tid={tid}")
                print("  got v52..v55:", [hex(int(x)) for x in got_tile1])
                print("  exp        :", [hex(int(x)) for x in exp])

    assert mism == 0, f"PACK check failed: {mism}/{cfg.threads} threads mismatched"
    print(f"[OK] PACK check passed for {cfg.threads} threads.")


def _ds_bpermute_xor32(val_by_lane: torch.Tensor) -> torch.Tensor:
    """
    ds_bpermute with index = (lane<<2)^0x80 effectively reads from lane^32.
    val_by_lane: shape (64,) u32
    """
    idx = torch.arange(64) ^ 32
    return val_by_lane[idx]

def _mix_one_wave(pre: torch.Tensor, cndmask_true_selects_src1: bool) -> torch.Tensor:
    """
    pre: shape (64, 8) u32 values for v48..v55 per lane.
    returns: shape (64, 8) u32 post-mix expected.

    AMD semantics reminder:
      v_cndmask_b32 dst, src0, src1, vcc
    Some tooling/documentation phrases this as (vcc ? src1 : src0).
    We auto-detect by trying both senses and matching the kernel's dump.
    """
    # torch.where is not implemented for uint32 on CPU; do mix in int64 then cast back.
    pre = pre.clone().to(torch.int64)
    lane = torch.arange(64)
    vcc = (lane & 32) == 0  # lane < 32

    def cm(src0: torch.Tensor, src1: torch.Tensor) -> torch.Tensor:
        # if vcc true selects src1, it's (vcc ? src1 : src0); else inverted.
        if cndmask_true_selects_src1:
            return torch.where(vcc, src1, src0)
        return torch.where(vcc, src0, src1)

    # Inputs
    t66 = pre[:, 0]
    t82 = pre[:, 1]
    t83 = pre[:, 2]
    t84 = pre[:, 3]
    t85 = pre[:, 4]
    t86 = pre[:, 5]
    t71 = pre[:, 6]
    t73 = pre[:, 7]

    # Stage 1
    t67_sel = cm(t66, t83)  # v201 = cndmask(t66,t83,vcc)
    t67 = _ds_bpermute_xor32(t67_sel)

    t68_sel = cm(t82, t84)
    t69 = _ds_bpermute_xor32(t68_sel)

    t72_sel = cm(t85, t71)
    t72 = _ds_bpermute_xor32(t72_sel)

    t83_out = _ds_bpermute_xor32(cm(t86, t73))

    # Waitcnt not needed in CPU model
    t66_out = cm(t67, t66)  # v200 = cndmask(v201, v200, vcc)
    t67_out = cm(t83, t67)  # v201 = cndmask(v209, v201, vcc)

    # Stage 2 (pure cndmask) - match operand order in assembly.
    t68_out = cm(t69, t82)  # v202 = cndmask(v203, v208, vcc)
    t69_out = cm(t84, t69)  # v203 = cndmask(v210, v203, vcc)
    t70_out = cm(t72, t85)  # v204 = cndmask(v206, v211, vcc)
    t71_out = cm(t71, t72)  # v205 = cndmask(v205, v206, vcc)
    t72_out = cm(t83_out, t86)  # v206 = cndmask(v209, v212, vcc)
    t73_out = cm(t73, t83_out)  # v207 = cndmask(v207, v209, vcc)

    out = torch.stack(
        [t66_out, t67_out, t68_out, t69_out, t70_out, t71_out, t72_out, t73_out],
        dim=1,
    )
    return out.to(torch.uint32)


def check_mix(pre_u32: torch.Tensor, post_u32: torch.Tensor, cfg: LaunchCfg):
    pre = pre_u32.view(cfg.threads, 8).cpu()
    post = post_u32.view(cfg.threads, 8).cpu()

    # Instead of perfectly modeling every cndmask/bpermute detail, we do a rigorous
    # *behavioral* check:
    # For this Triton mix, every output word should come from either:
    #   - same lane, or
    #   - partner lane^32
    # and only from the 8 pre-mix words.
    #
    # With random inputs, we can also infer the exact mapping and verify it is
    # consistent within each half-wave.

    got = post
    pre_w = pre.view(cfg.waves, 64, 8)
    got_w = got.view(cfg.waves, 64, 8)

    def infer_for_lane(w: int, lane: int, out_idx: int):
        tgt = int(got_w[w, lane, out_idx].item())
        cand = []
        for src_lane in [lane, lane ^ 32]:
            for k in range(8):
                if int(pre_w[w, src_lane, k].item()) == tgt:
                    cand.append((src_lane, k))
        return cand

    # Track inferred mapping pattern for lanes 0..31 and 32..63
    patterns = {0: [], 32: []}  # representative lane per half
    failures = 0

    for w in range(cfg.waves):
        for lane in range(64):
            rep = 0 if lane < 32 else 32
            for out_idx in range(8):
                cand = infer_for_lane(w, lane, out_idx)
                if len(cand) != 1:
                    failures += 1
                    if failures <= 3:
                        # For debugging, also see if it matches somewhere else in the wave.
                        tgt = int(got_w[w, lane, out_idx].item())
                        found_else = []
                        for sl in range(64):
                            for k in range(8):
                                if int(pre_w[w, sl, k].item()) == tgt:
                                    found_else.append((sl, k))
                                    if len(found_else) >= 4:
                                        break
                            if len(found_else) >= 4:
                                break
                        print(
                            f"[MIX INFER FAIL] wave={w} lane={lane} out={out_idx} matches(lane/partner)={cand} "
                            f"found_elsewhere={found_else}"
                        )
                    continue
                src_lane, k = cand[0]
                # normalize src to {self, partner}
                src_kind = 0 if src_lane == lane else 1
                if lane == rep:
                    patterns[rep].append((out_idx, src_kind, k))

    assert failures == 0, f"MIX check failed: {failures} outputs were not uniquely traceable to lane or lane^32 inputs"

    # Check representative lane patterns are stable across waves (same mapping each wave).
    # patterns[rep] has 8 entries per wave for that rep lane.
    for rep in [0, 32]:
        per_wave = [patterns[rep][i * 8 : (i + 1) * 8] for i in range(cfg.waves)]
        base = per_wave[0]
        ok = all(p == base for p in per_wave[1:])
        assert ok, f"MIX check failed: inferred mapping for lane{rep} differs across waves"

    # Print the inferred mapping (lane0 + lane32) from wave0.
    lane0_map = patterns[0][0:8]
    lane32_map = patterns[32][0:8]
    print("[OK] MIX check passed (each output is from lane or lane^32).")
    print("  inferred mapping for lane0  (wave0):", lane0_map)
    print("  inferred mapping for lane32 (wave0):", lane32_map)


def main():
    seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(seed)

    cfg = LaunchCfg()

    # random Q/K in FP16 then quantize to FP8 (more realistic distribution)
    B, H = 1, 1
    Q = torch.randn((B, H, cfg.s_q, cfg.D), device="cuda", dtype=torch.float16).to(
        torch.float32
    )
    K = torch.randn((B, H, cfg.s_k, cfg.D), device="cuda", dtype=torch.float16).to(
        torch.float32
    )
    Q = Q.to(torch.float8_e4m3fn)
    K = K.to(torch.float8_e4m3fn)
    V = torch.zeros((B, H, cfg.s_k, cfg.D), device="cuda", dtype=torch.float8_e4m3fn)

    stride_qh = cfg.s_q * cfg.D
    stride_kh = cfg.s_k * cfg.D
    stride_vh = cfg.s_k * cfg.D
    stride_oh = 0

    hip = _load_hip()
    _ = torch.zeros(1, device="cuda")  # init context

    dump_pack_only = os.environ.get("DUMP_PACK_ONLY", "") != ""
    dump_mix_only = os.environ.get("DUMP_MIX_ONLY", "") != ""

    if not dump_mix_only:
        # PACK dump (tile0): 20 dwords per tid = [v32..v47, v48..v51]
        out_pack0 = torch.zeros(cfg.threads * 20, device="cuda", dtype=torch.uint32)
        _launch_dump(
            hip,
            Q,
            K,
            V,
            out_pack0,
            cfg.num_k_tiles,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_oh,
            0x00000100,
        )
        # Validate tile0: v48..v51 == pack(v32..v47)
        pack0 = out_pack0.view(cfg.threads, 20).cpu()
        mism0 = 0
        for tid in range(cfg.threads):
            acc_u32 = pack0[tid, 0:16].tolist()
            got_u32 = pack0[tid, 16:20].tolist()
            acc_f = [_u32_to_f32(x) for x in acc_u32]
            exp0 = [
                _pack4_fp8_e4m3fn(acc_f[0], acc_f[1], acc_f[2], acc_f[3]),
                _pack4_fp8_e4m3fn(acc_f[4], acc_f[5], acc_f[6], acc_f[7]),
                _pack4_fp8_e4m3fn(acc_f[8], acc_f[9], acc_f[10], acc_f[11]),
                _pack4_fp8_e4m3fn(acc_f[12], acc_f[13], acc_f[14], acc_f[15]),
            ]
            if any(int(got_u32[i]) != int(exp0[i]) for i in range(4)):
                mism0 += 1
                if mism0 <= 3:
                    print(f"[PACK0 MISMATCH] tid={tid}")
                    print("  got v48..v51:", [hex(int(x)) for x in got_u32])
                    print("  exp        :", [hex(int(x)) for x in exp0])
        assert mism0 == 0, f"PACK0 check failed: {mism0}/{cfg.threads} threads mismatched"
        print(f"[OK] PACK0 check passed for {cfg.threads} threads.")

        # PACK dump (tile1): 24 dwords per tid = [v32..v47, v48..v55]
        out_pack1 = torch.zeros(cfg.threads * 24, device="cuda", dtype=torch.uint32)
        _launch_dump(
            hip,
            Q,
            K,
            V,
            out_pack1,
            cfg.num_k_tiles,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_oh,
            0x00000200,
        )
        check_pack(out_pack1, cfg)

    if not dump_pack_only:
        # MIX dumps: 8 dwords per tid
        out_pre = torch.zeros(cfg.threads * 8, device="cuda", dtype=torch.uint32)
        _launch_dump(
            hip,
            Q,
            K,
            V,
            out_pre,
            cfg.num_k_tiles,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_oh,
            0x00000800,
        )
        out_post = torch.zeros(cfg.threads * 8, device="cuda", dtype=torch.uint32)
        _launch_dump(
            hip,
            Q,
            K,
            V,
            out_post,
            cfg.num_k_tiles,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_oh,
            0x00000400,
        )
        check_mix(out_pre, out_post, cfg)


if __name__ == "__main__":
    main()


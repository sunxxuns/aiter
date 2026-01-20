# FP8 Scaffold Debug Status (2026-01-20)

Goal: document current kernel state, known-good parts, and open issues so we can resume without re-deriving context.

## Files in play
- `hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.s`
- `hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_qk_debug.s`
- `hsa/gfx950/fmha_v3_fwd_fp8/test_qk_debug.py`
- `hsa/gfx950/fmha_v3_fwd_fp8/test_scaffold_numerics.py`
- `hsa/gfx950/fmha_v3_fwd_fp8/triton_fp8_fmha.s` (reference)

## Current kernel layout (scaffold)
- Q: pitch-132 LDS layout (row = tid>>1, col = (tid&1)*64), 2 tiles in LDS.
  - Q global load uses v32–v47 to avoid clobbering `lane_id` (v10) and `wave_id` (v9).
- K: row-major LDS, loaded by **explicit** `buffer_load_dwordx4` + `ds_write_b128`
  - Mask: `tid < 256`
  - LDS addr: `K_LDS{0,1} + (tid & 255) * 16`
  - Rationale: `buffer_load ... offen lds` with vaddr is invalid/unstable.
- V: swizzled LDS store (bitop3:0x78), TR8 reads in PV using bitop3:0x36 base + XOR variants.
- QK MFMA: `v_mfma_f32_32x32x64_f8f6f4`
- P packing: `v_cvt_pk_fp8_f32` + mask + `v_perm_b32` (Triton-style `v_cvt_scalef32_pk_fp8_f32` was tried and reverted).
- Half-wave P permute: currently **removed** (perf baseline); rowid still incorrect.

## Known-good parts
- QK debug (`fwd_fp8_qk_debug.s`) passes non-uniform random gate:
  - `DEBUG_RANDOM_NONUNIFORM=1` uses scaled inputs (0.05 * rand + 0.02)
  - Gate thresholds: max<=1.0, mean<=0.05, corr>=0.95
  - This validates pitch-132 Q + row-major K address mapping in the debug kernel.

## Current failures (scaffold)
Identity-P numerics:
- `NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=col`
  - **PASS** in decoded output for small block (rows/cols match).
- `NUMERICS_V_PATTERN=rowid`
  - **FAIL**: row permutation (even-row pattern); not a simple transpose.

Interpretation: PV path still wrong for rowid; likely **P-to-A layout/transpose** mismatch, possibly V TR8 base.

## Perf note
- Current benchmark ~33.3 ms (~0.65 PF eq) because GPU0 sclk is stuck at ~150MHz (DPM level 1). Attempts to raise sclk via `rocm-smi` failed (manual/perf determinism not permitted).

## Recent adjustments (summary)
- **K preload rewrite**: replaced `buffer_load ... offen lds` with
  - `buffer_load_dwordx4 v[20:23], v_offset, s[12:15], 0 offen`
  - `ds_write_b128` into K_LDS0/1
  - This fixed the K preload reliability but did not fix PV numerics.
- **P packing**: reverted to `v_cvt_pk_fp8_f32` + mask + `v_perm_b32`.
- **Q load fix**: moved Q loads to v32–v47 to preserve `lane_id`/`wave_id`; QK identity now correct inside scaffold.
- **P permute**: several `ds_bpermute` variants tried; adjacent-pair permute keeps col-pattern correct but rowid still wrong.

## Repro commands
Build:
```
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mno-xnack \
  -c hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.s -o hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.o && \
/opt/rocm/llvm/bin/ld.lld -shared hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.o \
  -o hsa/gfx950/fmha_v3_fwd_fp8/fwd_fp8_scaffold.co
```

QK debug gate:
```
DEBUG_RANDOM_NONUNIFORM=1 python hsa/gfx950/fmha_v3_fwd_fp8/test_qk_debug.py
```

Identity-P PV checks:
```
NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=col python hsa/gfx950/fmha_v3_fwd_fp8/test_scaffold_numerics.py
NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=rowid python hsa/gfx950/fmha_v3_fwd_fp8/test_scaffold_numerics.py
```

## Next focus (priority)
1. **Derive correct P-to-A transpose**:
   - Use LDS transpose or port Triton’s exact `ds_bpermute`/`cndmask` sequence.
2. **Verify TR8 V base vs Triton**:
   - Compare `bitop3:0x36 + XOR` base set and `lane_id` usage.
3. **PV-only debug path**:
   - Dump packed P (v48–v55) and/or V TR8 reads to validate operand layouts.


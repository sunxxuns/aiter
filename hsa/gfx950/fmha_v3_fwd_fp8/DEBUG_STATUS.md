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
- K: row-major LDS, loaded by **explicit** `buffer_load_dwordx4` + `ds_write_b128`
  - Mask: `tid < 256`
  - LDS addr: `K_LDS{0,1} + (tid & 255) * 16`
  - Rationale: `buffer_load ... offen lds` with vaddr is invalid/unstable.
- V: swizzled LDS store (bitop3:0x78), TR8 reads in PV using bitop3:0x36 base + XOR variants.
- QK MFMA: `v_mfma_f32_32x32x64_f8f6f4`
- P packing: `v_cvt_pk_fp8_f32` + mask + `v_perm_b32` (Triton-style `v_cvt_scalef32_pk_fp8_f32` was tried and reverted).
- Half-wave P permute: currently **removed** (multiple ds_bpermute variants tried).

## Known-good parts
- QK debug (`fwd_fp8_qk_debug.s`) passes non-uniform random gate:
  - `DEBUG_RANDOM_NONUNIFORM=1` uses scaled inputs (0.05 * rand + 0.02)
  - Gate thresholds: max<=1.0, mean<=0.05, corr>=0.95
  - This validates pitch-132 Q + row-major K address mapping in the debug kernel.

## Current failures (scaffold)
Identity-P numerics still **FAIL** (column collapse / row-constant outputs):
- `NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=col`
  - Example: row values ~[0, 0.4375, 0.875, 1.3125, ...], **constant across columns**
- `NUMERICS_V_PATTERN=rowid`
  - Output is a large constant per row (e.g. ~1744), not the expected row ramp

Interpretation: PV path is still wrong. Column collapse strongly suggests a **P layout / permute mismatch** or **TR8 V base mismatch**.

## Recent adjustments (summary)
- **K preload rewrite**: replaced `buffer_load ... offen lds` with
  - `buffer_load_dwordx4 v[20:23], v_offset, s[12:15], 0 offen`
  - `ds_write_b128` into K_LDS0/1
  - This fixed the K preload reliability but did not fix PV numerics.
- **P packing**: reverted to `v_cvt_pk_fp8_f32` + mask + `v_perm_b32`.
- **P permute**: tried Triton-style `ds_bpermute` (using `v141 = tid << 2`, `v158 = v141 ^ 0x80`) and a simpler lane-permute. Neither fixed PV; some variants zeroed output.

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
1. **Reproduce Triton’s P permute exactly**:
   - In Triton, `v141 = tid << 2`, `v158 = v141 ^ 0x80`, and `ds_bpermute` is applied
   - Ensure cndmask/permute order matches `triton_fp8_fmha.s` lines ~635–690
2. **Verify TR8 V base**:
   - Compare our `bitop3:0x36 + XOR` base set with Triton’s base for PV reads
   - Confirm `tid` vs `lane_id` usage for V base
3. **Introduce a PV-only debug path**:
   - Dump packed P (v48–v55) or TR8 V reads to output and compare to reference
4. **Re-validate QK inside scaffold after K preload rewrite**
   - Optional: add a temporary early-exit that dumps v32–v47 and check against reference


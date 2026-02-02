---
name: amd-tr8-layout-debug
description: Debug and validate AMD CDNA MFMA TR8 (ds_read_*_tr_b8) LDS layouts and address generation in hand-written ASM kernels. Use when identity-P/rowcol tests fail, when TR8 coverage folds/misses rows, or when V LDS write swizzle/row-permute must be reconciled with TR8 read base/offsets and MFMA operand packing.
---

# AMD TR8 Layout Debug (MFMA + LDS)

## When to use

Use this skill when working on `fwd_fp8_scaffold.s` / FMHA ASM kernels and you see:

- `NUMERICS_IDENTITY_P=1` fails (especially `V_PATTERN=rowcol`)
- `V_ROWID` passes but `rowcol` fails (likely column/col-block packing issue)
- Raw TR8 reads miss rows / “fold by 8/16/32” patterns (likely row-permute/base mismatch)
- Perf work depends on enabling V swizzle/permute but correctness regresses

## Core mental model (keep these distinct)

- **LDS write layout (V staging)**: decides *where bytes live in LDS* after global loads.
- **TR8 read (`ds_read_*_tr_b8`)**: given an LDS address, produces a fixed MFMA-fragment interpretation of bytes.
- **TR8 base/offset generation**: decides *which permuted bytes are read* for a logical (k,row,col) tile.
- **MFMA B packing (v0..v7)**: decides how 32 FP8 bytes are arranged into 8 dwords per lane for MFMA.

Correctness requires the **pair**:
1) V LDS write layout, and
2) TR8 base/offsets (+ any row-permute inverse),
to be mutually consistent, and then MFMA packing must match the ISA dense layout expectations.

## “Bespoke layouts” vs our ASM approach (Triton blog comparison)

Reference: [“Triton Bespoke Layouts”](https://www.lei.chat/posts/triton-bespoke-layouts/).

- **Triton approach (compiler)**:
  - Layouts are carried in the MLIR type system (blocked/shared/MMA/dot-operand).
  - The compiler inserts `convert_layout` ops, coalesces global loads, chooses shared swizzles/padding, and lowers dot operands into the vendor MFMA layouts.
  - “Special layout” and “special load” are coordinated by compiler passes; you don’t hand-author LDS addresses/permutes.

- **Our approach (hand-written ASM)**:
  - We explicitly implement what Triton’s passes would have decided: LDS swizzles/permutes, TR8 base math, and MFMA operand packing.
  - We need solver-style tooling because the mapping is fragile and correctness is per-lane/per-wave, not statistical.
  - Debugging is done by dumping raw TR8 regs and packed operands and comparing against deterministic V patterns.

## Canonical validation workflow (identity-P)

### Step 0: Pick the right V patterns

- **Rowbyte**: `V[k, *] = k` (bytes 0..63 are unique row IDs).
  - Best for detecting row folding/missing rows.
- **Rowxorcol**: `V[k, c] = k ^ c` (disambiguates raw-byte source positions).
  - Best for mapping packed bytes back to raw TR8 positions.
- **Rowcol**: `V[k, c] = k + c/128` (or similar).
  - Best for final end-to-end correctness (detects column/col-block mismatches).

### Step 1: Prove whether row-permute is the culprit

Run raw TR8 dump with and without row permutation:

- **raw TR8 regs dump**: `debug_flags |= 0x01000000`
- **disable V row permutation**: `debug_flags |= 0x00000080`

Expected outcome:
- If `permOFF` becomes full coverage but `permON` misses rows → TR8 base/offsets do not compensate the V row permute.

Helpful script(s):
- `hsa/gfx950/fmha_v3_fwd_fp8/tools/dump_scaffold_tr8_raw_and_packed.py`

### Step 2: Validate MFMA B packing for each col block

With `V=rowbyte` and `NUMERICS_IDENTITY_P=1`, dump packed MFMA B bytes (`v0..v7`) and compare to ISA expectation:

- lanes 0..31 should be: `[0..15, 32..47]`
- lanes 32..63 should be: `[16..31, 48..63]`

Debug dumps available in scaffold:
- **B0**: dump PV operands before first PV MFMA: `debug_flags |= 0x00000001` (includes B=v0..v7)
- **B1**: dump B after col-block1 pack: `debug_flags |= 0x00004000` (repurposed)

Interpretation:
- If B0 is correct with `permOFF` but B1 (and later blocks) are wrong → the bug is in **block1/2/3 pack logic**, not TR8 itself.

### Step 3: If block1/2/3 pack is wrong, solve it deterministically

Goal: map each packed byte position `p0..p31` to a unique raw TR8 byte position `r` (within the 128B raw dump) for each lane group.

Procedure:
- Dump **raw TR8 regs** for 2–3 different deterministic patterns (rowbyte + rowxorcol (+ rowaddcol if needed)).
- For each thread (tid) build signature tuples per raw byte and per packed byte.
- Match signatures to deduce `packed[p] -> raw[r]`.
- Generate the MFMA B packing code from this mapping; do not guess.

### Step 4: Final correctness checks

Run, in this order:
- `NUMERICS_IDENTITY_P=1` with `V_ROWID=1` (row selection)
- `NUMERICS_IDENTITY_P=1` with `V_PATTERN=rowcol` (full row+col)
- exact-zero guards:
  - `NUMERICS_ZERO_Q=1`
  - `NUMERICS_ZERO_K=1`
  - `NUMERICS_ZERO_V=1`

## Practical notes

- TR8 correctness is **per-lane**; never accept “mean error small” as proof of layout correctness.
- If `V_ROWID` passes but `rowcol` fails: rows are probably correct but columns/col-block packing is wrong.
- When changing any of:
  - V LDS write swizzle,
  - V row permutation,
  - TR8 base/offset math,
you must revalidate TR8 raw coverage and packed B per col block.

## New verified fixes + repros (scaffold, gfx950)

This section contains **only verified facts** from runs in this repo (no speculation).

### 1) Fix: cross-iteration corruption due to clobbered `v61` (prefetch addressing)

Symptom:
- `NUMERICS_IDENTITY_P=1` with `NUMERICS_V_RANGE=64:128` produced **non-zero** output in rows `0..63` (should be all-zero if only V rows 64..127 are nonzero).

Root cause:
- `PV_PREFETCH_START` used `v61` (computed earlier as `(tid&255)*16`) after the PV/TR8 path had clobbered it, so **K/V prefetch addressing was wrong** across K-pair iterations.

Fix (in `fwd_fp8_scaffold.s`):
- Recompute `v61 = (tid & 255) * 16` at `PV_PREFETCH_START` before any global offset math / LDS writes.

Verification:
- After the fix, both splits pass:
  - `NUMERICS_V_RANGE=0:64` → exact match
  - `NUMERICS_V_RANGE=64:128` → exact match

### 2) Fix: identity-P `rowcol` required the v_read_dump-style TR8 base family (bitop3:0x36 + XOR ladder)

Symptom:
- `NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=rowcol NUMERICS_K_TILES=4` showed deterministic col-half errors (e.g. `+0.25` = `32/128`) even when row selection looked correct.

Root cause:
- PV TR8 base computation defaulted to a legacy heuristic path unless a debug flag was set. That produced the **wrong TR8 base family** relative to the LDS swizzle/write scheme.

Fix (in `fwd_fp8_scaffold.s`):
- Make PV TR8 base computation always follow the **v_read_dump-style** base (final seed via `v_bitop3_b32 ... bitop3:0x36`, then fixed XOR ladder and base-add/xor knobs from args with defaults 0).

Verification:
- Identity-P numerics become exact-zero error:
  - `NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=rowid NUMERICS_S=64 NUMERICS_K_TILES=4` → `max_err=0`
  - `NUMERICS_IDENTITY_P=1 NUMERICS_V_PATTERN=rowcol NUMERICS_S=64 NUMERICS_K_TILES=4` → `max_err=0`
  - plus the `V_RANGE` splits above.

### 3) Repro: one-command scaffold TR8 evidence log (diffable)

Files:
- Runner: `hsa/gfx950/fmha_v3_fwd_fp8/repro_scaffold_tr8.py`
- Doc: `hsa/gfx950/fmha_v3_fwd_fp8/REPRO_SCAFFOLD_TR8.md`
- Underlying dump tool: `hsa/gfx950/fmha_v3_fwd_fp8/tools/dump_scaffold_tr8_raw_and_packed.py`

Purpose:
- Build the `.co`, run deterministic dumps, and write a single log under:
  - `hsa/gfx950/fmha_v3_fwd_fp8/out/repro_scaffold_tr8/*.log`

### 4) Repro: per-group TR8 (k,col) decoder (compiler-like “dot operand” insight)

File:
- `hsa/gfx950/fmha_v3_fwd_fp8/tools/dump_scaffold_tr8_group_pairs.py`

What it does:
- Runs two passes with byte-coded V:
  - row-coded: `Vbytes[r,c]=r+1`
  - col-coded: `Vbytes[r,c]=c+1`
- Uses the group-selectable in-kernel dump to reconstruct per-byte `(k,col)` pairs for each TR8 group.

Use:
- This is the fastest way to see **col-half toggles** (col vs col+32) and validate whether a base/offset family matches the LDS write scheme.

## Confirmed learnings from Triton 3.4.0 (gfx950) dumps

These are **confirmed from actual generated artifacts** (not speculation) using:

- Script: `triton/experiments/amd_layout_dump/dump_mfma_lds_tr8.py`
- Env: `TRITON_KERNEL_DUMP=1 AMDGCN_ENABLE_DUMP=1 TRITON_DUMP_DIR=...`
- Outputs:
  - `mm32x32x64_fp16.ttgir` (layout-annotated IR)
  - `mm32x32x64_fp16.llir` (LLVM IR)
  - `mm32x32x64_fp16.amdgcn` (assembly)

### Bench + correctness (this environment)

This script also runs a correctness check and microbench:

- **Correctness**: compares kernel output `C` to `torch.matmul(A,B)` in fp16 and prints `max_err_fp16`.
- **Microbench**: uses `triton.testing.do_bench` and prints `avg_ms`.

Ground-truth from a run on this machine (copy/paste output):

- `max_err_fp16 = 0.0`
- `avg_ms ≈ 0.00663` (tiny 32x32x64 kernel; not meaningful for roofline)

### 1) Triton expresses the exact “bespoke layout → ASM” chain we emulate manually

In TTGIR, a matmul lowers as:

- `amdgpu.buffer_load` (global → regs)
- `ttg.local_alloc` (regs → LDS) with a **swizzled_shared** encoding
- `ttg.local_load` (LDS → regs) converting to **dot operand layout** (`#ttg.dot_op`)
- `tt.dot` using `#ttg.amd_mfma` encoding (MFMA)

So the compiler is doing the same conceptual steps we are hand-coding in ASM: choose a shared layout, then use TR loads to satisfy MFMA operand layout.

### 2) MFMA operands: kWidth=8 implies wider LDS reads and 4 MFMA steps for K=64

TTGIR shows `#ttg.dot_op{..., kWidth = 8}` for both A and B. In the generated assembly for `mm32x32x64_fp16`, you see:

- A operand loaded via four `ds_read_b128` into `v[2:5]`, `v[10:13]`, `v[14:17]`, `v[18:21]` (covering K=64 in chunks).
- B operand loaded via `ds_read_b64_tr_b16` (TR16 transpose-load) into `v[6:9]`, then additional `ds_read_b64_tr_b16` with offsets `1024/2048/3072` to cover the K dimension.
- Four MFMA ops:
  - `v_mfma_f32_32x32x16_f16` repeated 4× to cover K=64.

Implication for our FP8 attention: if we want “one MFMA consumes multiple K slices” we must coordinate kWidth-like packing with the LDS read pattern and MFMA schedule.

### 3) Shared swizzle shows up as concrete bitop3/xor address math in ASM

In `mm32x32x64_fp16.amdgcn`, the shared layout manifests as:

- Address generation using `v_bitop3_b32 ... bitop3:0x36` plus xors like `v_xor_b32_e32 v16, 0x88, ...` and `v_xor_b32_e32 v17, 0x288, ...`.
- LDS store addresses for A/B are derived from lane-id bits (m/n/k coordinates) and then XORed / mixed to implement the swizzled_shared encoding.

Implication: “shared swizzle” is not a semantic layer at runtime; it is literally the address bit-mixing we are currently hand-writing (`v_bitop3`, xor constants, masks).

### 4) TR loads are used directly for the MFMA dot operand layout

For B (fp16 case), Triton uses:

- `ds_read_b64_tr_b16` (TR16) as the primary mechanism to deliver the B fragment in MFMA-expected register arrangement.

Implication: our FP8 path’s TR8 reads are the analogous mechanism, and correctness hinges on the same two-part contract:

- V/K are written into LDS using a specific swizzle/permutation, and
- TR base/offset math must point at those permuted bytes so TR loads return the intended logical tile.



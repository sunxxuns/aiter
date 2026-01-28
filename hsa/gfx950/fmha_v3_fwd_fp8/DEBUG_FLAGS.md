## FP8 scaffold debug flags

The `fwd_fp8_scaffold.s` kernel takes an extra `debug_flags` `u32` argument (passed from:
`NUMERICS_DEBUG_FLAGS` in `test_scaffold_numerics.py` and `SCAFFOLD_DEBUG_FLAGS` in `test_scaffold.py`).

These flags are intended for **surgical, reproducible** debugging (dump-and-exit, operand isolation, etc).

### Flags (current)

- **`0x00000004`**: Use the `fwd_fp8_v_read_dump.s`-compatible **PV TR8 base path** driven by the extra `v_read_*` kernel args.
  - This is the preferred mode for solver/bruteforce, because it makes the scaffold and dump kernel share the same knobs.
- **`0x00000100`**: Dump **tile0** QK FP32 accumulators (`v32..v47`) plus packed tile0 P regs (`v48..v51`) and exit.
- **`0x00000200`**: Dump **tile1** QK FP32 accumulators (`v32..v47`) plus packed regs (`v48..v55`) and exit.
- **`0x00000400`**: Dump **post-mix** B operand regs (`v48..v55`) and exit.
- **`0x00000800`**: Dump **pre-mix** packed P regs (`v48..v55`) and exit.
- **`0x00001000`**: Dump tile1 QK FP32 accumulators (plus a few address regs) and exit.
- **`0x00010000`**: Force V bytes written to LDS to FP8(1.0) (`0x38`) before TR8 reads (A-isolation for PV).
- **`0x00020000`**: Dump B operand regs (`v48..v55`) and exit (older path; kept for compatibility).
- **`0x00080000`**: **Disable V LDS write swizzle** (identity write) for TR8 mapping experiments.
- **`0x00400000`**: Select solver-derived V-write swizzle (`bitop3:0x7a, C=0`) instead of baseline (`bitop3:0x78`).
- **`0x01000000`**: Dump raw TR8 read regs (`v200..v231`) and exit (used to prove A-isolation failures).
- **`0x02000000`**: Dump selected packed **V→A** regs (`v48..v55`, 32 bytes) and exit.
- **`0x04000000`**: Dump **all four lane-group** packed V→A groups and the selected one, then exit.
- **`0x20000000`**: Enable the **legacy P→A remap** path (debug-only; must be OFF for real PV correctness).
- **`0x80000000`**: Dump **MFMA operands** and exit (older combined debug path).

### ISA notes (TR8)

From `amd-instinct-cdna4-instruction-set-architecture.txt` §11.4 (“MFMA Transpose Load from LDS”):

- **EXEC must be all-ones** before `DS_READ_*_TR_*` instructions.
- **LDS address must be aligned** to the data size (`DS_READ_B64_TR_B8` requires 8-byte alignment).
- `DS_READ_B64_TR_B8` is designed to be used in **pairs** to load a complete matrix; the “first” instruction returns K
  buckets like `(0..7, 16..23, 32..39, 48..55)` and the “second” returns the complementary K values.

### Rigorous random-B validation

Use `tools/test_b_operand_random.py` to validate the **B path** under random inputs:

- **PACK correctness**: compares dumped FP32 QK accumulators against packed FP8 bytes.
- **MIX correctness**: dumps pre-mix/post-mix and verifies the output is a stable permutation from lane or lane^32.

Example:

```bash
SEED=123 python3 hsa/gfx950/fmha_v3_fwd_fp8/tools/test_b_operand_random.py
```


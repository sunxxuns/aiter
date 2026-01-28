## FP8 scaffold debug flags

The `fwd_fp8_scaffold.s` kernel takes an extra `debug_flags` `u32` argument (passed from:
`NUMERICS_DEBUG_FLAGS` in `test_scaffold_numerics.py` and `SCAFFOLD_DEBUG_FLAGS` in `test_scaffold.py`).

These flags are intended for **surgical, reproducible** debugging (dump-and-exit, operand isolation, etc).

### Flags (current)

- **`0x00000100`**: Dump **tile0** QK FP32 accumulators (`v32..v47`) plus packed tile0 P regs (`v48..v51`) and exit.
- **`0x00000200`**: Dump **tile1** QK FP32 accumulators (`v32..v47`) plus packed regs (`v48..v55`) and exit.
- **`0x00000400`**: Dump **post-mix** B operand regs (`v48..v55`) and exit.
- **`0x00000800`**: Dump **pre-mix** packed P regs (`v48..v55`) and exit.
- **`0x00001000`**: Dump tile1 QK FP32 accumulators (plus a few address regs) and exit.
- **`0x00010000`**: Force V bytes written to LDS to FP8(1.0) (`0x38`) before TR8 reads (A-isolation for PV).
- **`0x00020000`**: Dump B operand regs (`v48..v55`) and exit (older path; kept for compatibility).
- **`0x00400000`**: Select solver-derived V-write swizzle (`bitop3:0x7a, C=0`) instead of baseline (`bitop3:0x78`).

### Rigorous random-B validation

Use `tools/test_b_operand_random.py` to validate the **B path** under random inputs:

- **PACK correctness**: compares dumped FP32 QK accumulators against packed FP8 bytes.
- **MIX correctness**: dumps pre-mix/post-mix and verifies the output is a stable permutation from lane or lane^32.

Example:

```bash
SEED=123 python3 hsa/gfx950/fmha_v3_fwd_fp8/tools/test_b_operand_random.py
```


## Repro: scaffold TR8 (deterministic)

This repro is designed to isolate **V LDS write → TR8 read** behavior in `fwd_fp8_scaffold.s`, without depending on full end-to-end numerics.

### What it does

- Builds `fwd_fp8_scaffold.co`
- Runs the scaffold in a one-block configuration
- Initializes V as **rowbyte** (V[k,*] = k, stored as raw FP8 bytes)
- Dumps:
  - raw TR8 regs (v200..v231) via `debug_flags=0x01000000`
  - packed A bytes (v48..v55) via `debug_flags=0x00200000`

### One-command run

```bash
cd /sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8
python3 repro_scaffold_tr8.py
```

Outputs a log file under:

- `aiter/hsa/gfx950/fmha_v3_fwd_fp8/out/repro_scaffold_tr8/*.log`

### Common knobs

These are forwarded into `tools/dump_scaffold_tr8_raw_and_packed.py`:

- `HIP_VISIBLE_DEVICES=0`
- `DUMP_IDENTITY_WRITE=1` (default; uses the scaffold “identity V write” mode)
- `DUMP_COLBLK=0`
- `DUMP_PERM_ID=0`
- `DUMP_WRITE_MODE=2`

Example: run col-block 1

```bash
cd /sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8
DUMP_COLBLK=1 python3 repro_scaffold_tr8.py
```

### Why this is useful

If rowbyte fails here (missing/duplicated row IDs in raw TR8, or low positional-match in packed bytes),
the issue is definitively in:

- LDS write layout and/or
- TR8 base/offset generation and/or
- packing/mapping logic (if you’re packing into MFMA operands)

…not in softmax, scaling, or downstream epilog.


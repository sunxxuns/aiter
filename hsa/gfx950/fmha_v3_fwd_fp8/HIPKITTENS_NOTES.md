# HipKittens Notes (TR8/LDS/Swizzle)

Source: `aiter/hipkittens.pdf` → `aiter/hipkittens.txt`

## Key Excerpts (Relevant to PV TR8 Issue)

### Swizzle + phase/bank behavior is instruction-dependent
```
Shared memory accesses result in bank conflicts if multiple threads in a wave attempt to access the same bank
simultaneously. Waves ... execute shared memory accesses in phases ... on AMD, the phases are non-sequential
and differ based on the memory instruction.
```
```
Shared memory on AMD CDNA4 GPUs have different banking behavior depending on the instruction. ds read b128
accesses shared memory through 64 banks ... The swizzle applied here swaps the first 8 columns with the last 8
starting from the 8th row. This swizzling strategy simultaneously enables bank-conflict free accesses from
column-major reads using ds read b64 tr b16.
```

### A single swizzle cannot satisfy different DS ops
```
A single swizzle is not possible ... ds_write_b64 ... requires a swizzle pattern that respects the phase ordering
and bank behavior ... offset ^= ((offset % 512) >> 7) << 3 ...
```
```
ds_read_b128 requires at least 128 bits of memory to be contiguous ... the swizzle pattern for ds_write_b64
breaks apart memory into 64-bit chunks. As a result, different swizzling patterns need to be used for each.
```

### When docs are missing, derive with solvers
```
Since per-instruction phase and bank behavior is not well documented, we create simple solvers for both.
The phase solver iterates over every pair of threads in a wave ... The bank solver ...
```

## Takeaways for Our PV/TR8 Debug
- TR8 layout/swizzle is **not documented**; must be derived empirically.
- A swizzle tuned for ds_write_b128 or ds_read_b128 will **not** automatically match TR8 reads.
- We should **solve the address coverage**: TR8 read address set must be fully covered by V write address set.

## Proposed Next Steps
1. Generate full read/write address coverage maps (all lanes).
2. Derive a V LDS write swizzle that **exactly covers** TR8 read address set.
3. Validate with TR8 raw dumps (block IDs 0–3 only for `NUMERICS_V_RAW=1`).

## Session Update (TR8 coverage)
- Added `tools/tr8_layout_solver.py` to compare TR8 reads vs V writes.
- Current formulas show **almost zero coverage**:
  - total_reads=768, covered=2 (overall)
  - lane0 dump read_addrs=12, covered=0
- Searching simple XOR swizzles and write offsets did **not** improve coverage.
- Using TR8-base for writes improves global coverage slightly but still misses most read addrs.

## Session Update (Swizzle Solver Loop)
- Added `tools/tr8_swizzle_solver.py` and `tools/tr8_offset_search.py`.
- Best search so far:
  - Linear base with shifts `(3,7)`, `base_xor=0x20`, `tid_mask=0x3`, `tid_shift=7`
  - Offsets `(0,256,512,768,1024,1152,1280,1408)`
  - Coverage ≈ `508 / 768` (still far from full)
- Implemented candidate write layout in `fwd_fp8_scaffold.s` (debug path).
  - TR8 raw dump still shows block IDs 4–7 → coverage insufficient.

# HipKittens Solver Notes

Source: `HipKittens/analysis/paper_experiments/phases/*`

## Phase Solver (ds_read_b64, ds_write_b64)
Files:
- `analysis/paper_experiments/phases/ds_read_b64/phase_solver.py`
- `analysis/paper_experiments/phases/ds_write_b64/phase_solver.py`

Approach:
- Uses `rocprofv3` counters `SQ_INSTS_LDS` and `SQ_LDS_BANK_CONFLICT`.
- Enables only two threads and forces both to access the same bank.
- A nonzero conflict implies the two threads are in the same phase.
- Runs all thread pairs to build a conflict matrix, then groups threads into phases.

## Bank Solver (ds_read_b64)
File:
- `analysis/paper_experiments/phases/ds_read_b64/bank_solver.py`

Approach:
- Pick two threads from the same phase.
- Thread 0 reads a fixed bank (offset 0).
- Thread 1 scans bank offsets; first conflict indicates bank wraparound.
- This derives the number of LDS banks for the instruction.

## Practical Swizzle Hint (attention kernel)
File:
- `training/llama/csrc/attn_bkwd_causal_HNB.cpp`

Observed swizzle:
```
lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
```

Takeaway:
- They use an XOR swizzle on byte offsets with `(offset >> 9) << 5`.
- This is consistent with the paper’s note: swizzle patterns are instruction-dependent and
  must be derived empirically for each LDS instruction.

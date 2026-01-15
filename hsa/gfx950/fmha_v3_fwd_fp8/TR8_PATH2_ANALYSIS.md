# Path 2: TR8-Compatible Layout Implementation

## CRITICAL FINDING: TR8 Does NOT Work Like Simple Stride-8 Gather!

### Test Results (2024-01-15):
```
LDS setup: LDS[tid*8] = tid for tid in 0..63
Individual reads: LDS[0]=0, LDS[8]=1 (CORRECT)
TR8 results:
  Thread 0: [0, 0, 0, 0, 0, 0, 0, 0]  (NOT [0,1,2,3,4,5,6,7])
  Thread 1: [48, 48, 48, 48, 48, 48, 48, 48]
  Thread 2: [53, 53, 53, 53, 53, 53, 53, 53]
```

### Analysis:
- TR8 is NOT a simple stride-8 gather
- Each thread gets 8 IDENTICAL values (broadcast pattern)
- Different threads get different values
- This suggests TR8 is a **cooperative cross-lane instruction** for MFMA

### Implications:
- The layout requirements for TR8 are MFMA-specific
- Cannot treat TR8 as a simple LDS gather instruction
- Need to understand MFMA's specific input data layout expectations
- **Path 2 may be significantly more complex than originally thought**

---

## Original (Incorrect) Assumption About TR8

~~`ds_read_b64_tr_b8` reads 8 bytes with **stride 8**:~~
```
~~byte 0 from LDS[base + 0]~~
~~byte 1 from LDS[base + 8]~~
~~...~~
```
This assumption was **WRONG**.

## Required LDS Layout

For Q[32×128] FP8, to read Q[row, k:k+8] with TR8:

### Interleaved-8 Layout
Store rows in interleaved blocks of 8:
```
Block 0 (rows 0-7):
  Q[0, k] at LDS[0 + k*8]
  Q[1, k] at LDS[1 + k*8]
  ...
  Q[7, k] at LDS[7 + k*8]

Block 1 (rows 8-15):
  Q[8, k]  at LDS[1024 + 0 + k*8]
  Q[9, k]  at LDS[1024 + 1 + k*8]
  ...
  Q[15, k] at LDS[1024 + 7 + k*8]

... (4 blocks total for 32 rows)
```

### Formula
```
LDS_offset(row, k) = (row % 8) + (row / 8) * 1024 + k * 8
```

### LDS Size
- Each block: 8 rows × 128 k-values × 8 stride = 1024 bytes
- 4 blocks for 32 rows = 4096 bytes
- Same total size as row-major! Just rearranged.

## TR8 Read Address Calculation

For lane `L` to read Q[mfma_row, k_start:k_start+8]:
```asm
// mfma_row from lane mapping
v_and_b32_e32 v10, 7, v_lane       // row_in_block = lane & 7
v_lshrrev_b32_e32 v11, 3, v_lane   // block_idx = lane >> 3
v_lshlrev_b32_e32 v11, 10, v11     // block_base = block_idx * 1024

// For k iteration k_iter (each covers 8 k values)
v_lshlrev_b32_e32 v12, 6, v_k_iter // k_base = k_iter * 64 (8 vals * 8 stride)

// Final address
v_add3_u32 v_addr, v10, v11, v12   // row_in_block + block_base + k_base
ds_read_b64_tr_b8 v[20:21], v_addr
```

## The Challenge: Creating This Layout

### Why buffer_load...lds Won't Work
`buffer_load_dwordx4 ... offen lds` writes **16 consecutive bytes** to LDS.
TR8 layout needs **scattered bytes** (stride 8).

### Solution: Load to VGPR, Scatter to LDS

```asm
// Step 1: Load 16 bytes from global to VGPR
buffer_load_dwordx4 v[0:3], s[8:11], v_goffset offen

// Step 2: Compute scatter addresses
// Thread t loads Q[row, k_chunk*16 : k_chunk*16+15]
// where row = t / 8, k_chunk = t % 8

// Scatter address for byte i: (row % 8) + (row / 8) * 1024 + (k_chunk*16 + i) * 8
v_lshrrev_b32_e32 v10, 3, v_tid    // row = tid / 8
v_and_b32_e32 v11, 7, v_tid        // k_chunk = tid % 8
v_and_b32_e32 v12, 7, v10          // row_in_block = row % 8
v_lshrrev_b32_e32 v13, 3, v10      // block = row / 8
v_lshlrev_b32_e32 v13, 10, v13     // block_base = block * 1024
v_lshlrev_b32_e32 v14, 7, v11      // k_base = k_chunk * 16 * 8 = k_chunk * 128
v_add3_u32 v_lds_base, v12, v13, v14

// Step 3: Scatter 16 bytes to LDS with stride 8
// Unroll for bytes 0-15
s_waitcnt vmcnt(0)

// Byte 0
v_bfe_u32 v20, v0, 0, 8
ds_write_b8 v_lds_base, v20

// Byte 1 -> offset +8
v_add_u32 v21, 8, v_lds_base
v_bfe_u32 v20, v0, 8, 8
ds_write_b8 v21, v20

// Byte 2 -> offset +16
v_add_u32 v21, 16, v_lds_base
v_bfe_u32 v20, v0, 16, 8
ds_write_b8 v21, v20

// ... continue for all 16 bytes
```

### Problem: ds_write_b8 is SLOW
Writing 16 bytes one at a time with `ds_write_b8` is terrible for performance.

### Better Approach: ds_write_b32 with Packing

Since we're writing with stride 8, we can pack 4 bytes from 4 different threads:
```
Thread 0 byte 0 -> LDS[0]
Thread 1 byte 0 -> LDS[1]
Thread 2 byte 0 -> LDS[2]
Thread 3 byte 0 -> LDS[3]
```
These 4 bytes form one dword at LDS[0:3]!

With wave permute operations, we can gather bytes from 4 threads and write as dword.

## Optimized TR8 Load Pattern

### Concept: Cooperative Scatter
8 threads cooperate to write one "column" of the interleaved layout:
- Thread 0 contributes byte for row 0
- Thread 1 contributes byte for row 1
- ...
- Thread 7 contributes byte for row 7

```asm
// Thread t in group of 8
v_and_b32_e32 v_row_in_group, 7, v_tid

// Get byte for this column
// Each thread extracts different byte from its loaded data
v_bfe_u32 v_my_byte, v0, v_byte_idx, 8

// Cross-lane shuffle to gather 8 bytes
ds_bpermute_b32 ...  // Gather bytes from group of 8

// Pack 8 bytes into 2 dwords
v_perm_b32 ...

// Write 8 bytes at once
ds_write_b64 v_lds_addr, v[packed:packed+1]
```

### Alternative: Use ds_write2_b32 with Offsets
```asm
// Write 2 dwords at stride-4 addresses
ds_write2_b32 v_base, v_data0, v_data1 offset0:0 offset1:4
```
But this still writes consecutive dwords, not stride-8 bytes.

## Performance Comparison

| Method | LDS Writes | Instructions | Conflicts |
|--------|------------|--------------|-----------|
| buffer_load...lds (pitch-136) | 1 per 16B | 1 | ~0 |
| Scatter ds_write_b8 | 16 per 16B | 16 | High |
| Cooperative ds_write_b64 | 2 per 16B | Complex shuffle | ~0 |

## Conclusion: Path 2 Complexity

TR8-compatible layout requires:
1. Load to VGPR (can't use buffer_load...lds directly)
2. Complex byte extraction and cross-lane shuffle
3. Scattered writes to LDS

**Trade-off:**
- TR8 reads are faster than ds_read_b64
- But creating TR8 layout is expensive

**Recommendation:**
Only worth it if:
- Read bandwidth is the bottleneck (many K iterations)
- The shuffle overhead is amortized over many reads

For QK computation with few K tiles, Path 1 (pitch-136 + ds_read_b64) is likely better.
For full attention with many K tiles, Path 2 might win.

## Test Plan for Path 2

1. Create minimal kernel with TR8 layout using ds_write scatter
2. Verify numerical correctness with structured input
3. Benchmark vs pitch-136 + ds_read_b64
4. If faster, optimize the scatter with cross-lane ops

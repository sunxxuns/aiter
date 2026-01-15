#!/usr/bin/env python3
"""
Small tests to verify Triton's layout ideas for FP8 MFMA.

Key question: How does Triton ensure write row == read row?
Answer: LinearLayout computes consistent thread-to-element mapping.
"""

import numpy as np

def mfma_lane_to_row(lane):
    """
    MFMA 32x32x16 FP8: Which row does lane L own in the A matrix?
    
    From AMD Matrix Calculator:
    A[i][k].block Lane: 32 * floor(k / 8) + i
    
    So lane L owns row (L % 32) for k in [0,7] and [8,15]
    """
    return lane % 32

def mfma_lane_to_col_base(lane):
    """
    MFMA 32x32x16 FP8: Which k-columns does lane L access?
    
    Lanes 0-31: k = 0-7
    Lanes 32-63: k = 8-15
    """
    return 0 if lane < 32 else 8

def our_write_row(tid):
    """
    Our LDS write mapping: tid -> row
    256 threads write 32 rows of 128 bytes each.
    Each row has 8 threads writing 16 bytes.
    """
    return tid // 8

def our_write_col(tid):
    """
    Our LDS write mapping: tid -> column offset
    """
    return (tid % 8) * 16

def test_row_mapping_mismatch():
    """
    Test 1: Show the write row != read row problem
    """
    print("=" * 70)
    print("TEST 1: Write Row vs Read Row Mismatch")
    print("=" * 70)
    
    print("\nFor 256 threads writing 32 rows:")
    print(f"{'TID':>4} | {'Write Row':>10} | {'MFMA Row':>10} | {'Match':>6}")
    print("-" * 40)
    
    mismatches = 0
    for tid in range(64):  # Show first 64 threads (1 wave)
        write_row = our_write_row(tid)
        mfma_row = mfma_lane_to_row(tid)
        match = "YES" if write_row == mfma_row else "NO"
        if write_row != mfma_row:
            mismatches += 1
        if tid < 16 or (tid >= 32 and tid < 48):  # Show subset
            print(f"{tid:>4} | {write_row:>10} | {mfma_row:>10} | {match:>6}")
    
    print(f"\nMismatches in first 64 threads: {mismatches}/64")
    print("\nConclusion: Write row and MFMA row are DIFFERENT mappings!")

def test_triton_linear_layout():
    """
    Test 2: Simulate Triton's LinearLayout for MFMA operand A
    
    From Triton's chooseDotDsReadTrLayout():
    - Register dimension: which elements thread 0 holds
    - Lane dimension: which elements are in register 0 of each lane
    """
    print("\n" + "=" * 70)
    print("TEST 2: Triton LinearLayout for MFMA 32x32x16 FP8")
    print("=" * 70)
    
    # Triton's register bases for mfma32 FP8 (from LinearLayoutConversions.cpp)
    # registerBase tells us which (row, col) element is in register bit N
    register_bases = [
        (1, 0),   # reg bit 0 -> row += 1
        (2, 0),   # reg bit 1 -> row += 2
        (4, 0),   # reg bit 2 -> row += 4
        (0, 16),  # reg bit 3 -> col += 16 (for k > 16)
    ]
    
    # laneBase tells us which element is in reg 0 for each lane
    lane_bases = [
        (0, 1),   # lane bit 0 -> col += 1
        (0, 2),   # lane bit 1 -> col += 2
        (0, 4),   # lane bit 2 -> col += 4
        (0, 8),   # lane bit 3 -> col += 8
        (8, 0),   # lane bit 4 -> row += 8
        (16, 0),  # lane bit 5 -> row += 16
    ]
    
    def compute_element(lane, reg):
        """Compute which (row, col) element is at (lane, reg)"""
        row, col = 0, 0
        # Apply lane bases
        for bit, (dr, dc) in enumerate(lane_bases):
            if lane & (1 << bit):
                row += dr
                col += dc
        # Apply register bases
        for bit, (dr, dc) in enumerate(register_bases):
            if reg & (1 << bit):
                row += dr
                col += dc
        return row, col
    
    print("\nTriton LinearLayout element mapping:")
    print(f"{'Lane':>4} | {'Reg 0':>12} | {'Reg 1':>12} | {'Reg 7':>12}")
    print("-" * 50)
    
    for lane in [0, 1, 2, 16, 32, 48, 63]:
        r0 = compute_element(lane, 0)
        r1 = compute_element(lane, 1)
        r7 = compute_element(lane, 7)
        print(f"{lane:>4} | {str(r0):>12} | {str(r1):>12} | {str(r7):>12}")
    
    print("\nKey insight: Lane determines (row, col) base, register adds row offset")
    print("This is DIFFERENT from our simple tid/8 write mapping!")

def test_diagonal_padding():
    """
    Test 3: Triton's diagonal padding pattern
    
    From Utility.cpp composePaddedLayoutForAsyncCopyCDNA4():
    Rows are arranged diagonally to avoid bank conflicts.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Triton Diagonal Padding Pattern")
    print("=" * 70)
    
    # Triton parameters for FP8
    vec_size = 8  # bytes per vector
    warp_size = 64
    padding_interval = vec_size * warp_size  # 512 bytes
    row_size = 128  # bytes per row
    k_width = 8  # FP8 kWidth for mfma32
    
    # wrap = how many rows before diagonal pattern repeats
    wrap = min(row_size, 128) // k_width  # = 16
    
    print(f"\nParameters: vec_size={vec_size}, padding_interval={padding_interval}")
    print(f"wrap={wrap} (pattern repeats every {wrap} rows)")
    
    print("\nDiagonal row arrangement:")
    print(f"{'Logical Row':>12} | {'Group':>6} | {'Pos in Group':>13} | {'Diag Offset':>11}")
    print("-" * 50)
    
    for row in range(32):
        group = row % wrap
        pos_in_group = row // wrap
        diag_offset = group * k_width  # Each group shifts by kWidth
        print(f"{row:>12} | {group:>6} | {pos_in_group:>13} | {diag_offset:>11}")
        if row == 15:
            print("..." + "-" * 47)

def test_triton_write_read_consistency():
    """
    Test 4: Show how Triton ensures write/read consistency
    
    The key is that BOTH write and read use the same LinearLayout-derived mapping.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Triton Write/Read Consistency")
    print("=" * 70)
    
    # Simulated Triton approach: use LinearLayout for both
    def triton_thread_to_element(tid, k_chunk):
        """
        Triton's mapping: thread -> (row, col) using LinearLayout
        This is used for BOTH write and read!
        """
        # Simplified: in reality this comes from LinearLayout composition
        lane = tid % 64
        # Lane determines row and col base
        row = (lane & 7) + ((lane >> 4) & 1) * 8 + ((lane >> 5) & 1) * 16
        col = ((lane >> 3) & 1) * 8 + k_chunk * 16
        return row, col
    
    print("\nTriton uses SAME mapping for write and read:")
    print(f"{'TID':>4} | {'Write (row,col)':>15} | {'Read (row,col)':>15} | {'Match':>6}")
    print("-" * 50)
    
    for tid in [0, 1, 8, 16, 32, 48, 63]:
        write_elem = triton_thread_to_element(tid, k_chunk=0)
        read_elem = triton_thread_to_element(tid, k_chunk=0)  # Same function!
        match = "YES" if write_elem == read_elem else "NO"
        print(f"{tid:>4} | {str(write_elem):>15} | {str(read_elem):>15} | {match:>6}")
    
    print("\nConclusion: Triton guarantees consistency by using same mapping!")
    print("Our approach uses DIFFERENT mappings (tid/8 vs mfma_row), causing issues.")

def test_our_approach_fix():
    """
    Test 5: How to fix our approach - use consistent mapping
    """
    print("\n" + "=" * 70)
    print("TEST 5: Fixing Our Approach")
    print("=" * 70)
    
    print("""
Option A: Change LDS write to use MFMA-compatible mapping
---------------------------------------------------------
Instead of: row = tid / 8 (simple, but wrong)
Use:        row = mfma_lane_to_row(tid % 64) (matches read)

But this is complex because:
- Need to handle 256 threads writing 32 rows
- Each row needs 128 bytes from multiple threads
- Thread cooperation pattern must match MFMA expectations

Option B: Use XOR swizzle (row-agnostic)
----------------------------------------
Write: addr = (row * 144 + col) XOR ((row * 4) & 0xFF)
Read:  addr = (mfma_row * 144 + k_off) XOR ((mfma_row * 4) & 0xFF)

Both use XOR on final address, so row mismatch doesn't matter!
The XOR just spreads addresses across banks.

Option C: Accept 2-way conflicts (current)
------------------------------------------
Pitch-136 gives 2-way conflicts (~22% overhead at scale).
Simpler code, acceptable performance.
""")

def test_bank_conflict_comparison():
    """
    Test 6: Compare bank conflicts between approaches
    """
    print("\n" + "=" * 70)
    print("TEST 6: Bank Conflict Comparison")
    print("=" * 70)
    
    def analyze_conflicts(name, addr_func):
        banks = {}
        for lane in range(64):
            addr = addr_func(lane)
            bank = (addr // 4) % 64
            if bank not in banks:
                banks[bank] = []
            banks[bank].append(lane)
        max_conf = max(len(v) for v in banks.values())
        return len(banks), max_conf
    
    approaches = [
        ("Pitch-128 (naive)", 
         lambda l: (l % 32) * 128 + (8 if l >= 32 else 0)),
        ("Pitch-136 (ours)", 
         lambda l: (l % 32) * 136 + (8 if l >= 32 else 0)),
        ("Pitch-144 + XOR", 
         lambda l: ((l % 32) * 144 + (8 if l >= 32 else 0)) ^ ((l % 32) * 4)),
        ("Triton diagonal", 
         lambda l: (l % 32) * 136 + (8 if l >= 32 else 0) + ((l % 32) % 16) * 8),
    ]
    
    print(f"{'Approach':<25} | {'Banks Used':>10} | {'Max Conflict':>12}")
    print("-" * 55)
    
    for name, func in approaches:
        num_banks, max_conf = analyze_conflicts(name, func)
        status = "ZERO" if max_conf == 1 else f"{max_conf}-way"
        print(f"{name:<25} | {num_banks:>10} | {status:>12}")

def main():
    test_row_mapping_mismatch()
    test_triton_linear_layout()
    test_diagonal_padding()
    test_triton_write_read_consistency()
    test_our_approach_fix()
    test_bank_conflict_comparison()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Triton Layout Insights:
1. LinearLayout ensures SAME thread-to-element mapping for write and read
2. Diagonal padding shifts rows to spread across banks
3. ds_read_tr8_b64 expects data in MFMA-compatible layout

Our Issue:
- Write uses tid/8 for row
- Read uses mfma_lane_to_row(tid%64) for row
- These are DIFFERENT, so diagonal padding fails

Solutions:
- XOR swizzle (row-agnostic, recommended)
- Full LinearLayout rewrite (complex)
- Accept pitch-136 with 2-way conflicts (current)
""")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Deep analysis of bank conflicts for FP8 QK MFMA.
Find optimal swizzle pattern for zero bank conflicts.
"""

import numpy as np
from collections import defaultdict

def mfma_row_mapping(lane):
    """MFMA A operand row mapping for 32x32x16"""
    row16 = (lane & 3) + ((lane >> 3) & 3) * 4
    row_hi = (lane >> 2) & 1
    return row16 + row_hi * 16

def xor_swizzle(offset):
    """Current XOR swizzle"""
    return offset ^ (((offset & 0x1ff) >> 7) << 3)

def analyze_phase_access():
    """Analyze which addresses each phase accesses"""
    print("ds_read_b64 Phase Analysis for QK MFMA")
    print("=" * 70)
    
    # ds_read_b64 executes in 4 phases, 16 lanes each
    # Each lane reads 8 bytes (2 banks worth)
    
    print("\nMFMA row mapping for lanes 0-15 (phase 0):")
    print("-" * 50)
    
    rows_phase0 = []
    for lane in range(16):
        row = mfma_row_mapping(lane)
        rows_phase0.append(row)
        print(f"  Lane {lane:2d}: row {row:2d}")
    
    print(f"\nUnique rows in phase 0: {sorted(set(rows_phase0))}")
    print(f"Row distribution: {len(set(rows_phase0))} unique out of 16")
    
    # For HD=128 row-major, row R starts at offset R*128
    # So rows in phase 0 start at: row * 128
    print("\nBase addresses (no swizzle) for phase 0:")
    addrs = [r * 128 for r in rows_phase0]
    banks = [(a // 4) % 32 for a in addrs]
    print(f"  Addresses: {addrs}")
    print(f"  Banks: {banks}")
    print(f"  Unique banks: {len(set(banks))} out of 16 needed")
    
    # The problem: multiple rows map to same bank
    bank_count = defaultdict(int)
    for b in banks:
        bank_count[b] += 1
    conflicts = sum(c - 1 for c in bank_count.values() if c > 1)
    print(f"  Bank conflicts: {conflicts}")

def find_optimal_swizzle():
    """Try to find a swizzle that eliminates bank conflicts"""
    print("\n\nSearching for Optimal Swizzle")
    print("=" * 70)
    
    # Get the rows accessed in phase 0
    rows = [mfma_row_mapping(lane) for lane in range(16)]
    base_addrs = [r * 128 for r in rows]  # K=0 column
    
    print(f"Rows accessed: {rows}")
    print(f"Base addresses: {base_addrs}")
    
    # For zero bank conflicts, we need 16 addresses to map to 16 different banks
    # Bank = (swizzled_addr // 4) % 32
    
    # Try different XOR patterns
    best_conflicts = 16
    best_pattern = None
    
    for xor_bits in range(16):  # Which bits to XOR from high to low
        for shift in range(8):  # How much to shift
            def test_swizzle(addr):
                high_bits = (addr >> 7) & 0xf  # bits 7-10
                xor_val = (high_bits & xor_bits) << shift
                return addr ^ xor_val
            
            swizzled = [test_swizzle(a) for a in base_addrs]
            banks = [(s // 4) % 32 for s in swizzled]
            unique = len(set(banks))
            conflicts = 16 - unique
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_pattern = (xor_bits, shift)
    
    print(f"\nBest XOR pattern found: xor_bits={best_pattern[0]:04b}, shift={best_pattern[1]}")
    print(f"Conflicts: {best_conflicts}")
    
    # Try row-based swizzle (add row-dependent offset)
    print("\n\nTrying Row-Based Swizzle:")
    print("-" * 50)
    
    for row_mult in [8, 16, 24, 32, 64, 128]:
        def row_swizzle(addr, row):
            return addr + (row * row_mult) % 512
        
        swizzled = [row_swizzle(base_addrs[i], rows[i]) for i in range(16)]
        banks = [(s // 4) % 32 for s in swizzled]
        unique = len(set(banks))
        conflicts = 16 - unique
        print(f"  row_mult={row_mult:3d}: {conflicts} conflicts, banks={banks[:8]}...")

def analyze_bf16_pattern():
    """Analyze what BF16 does for bank-conflict-free access"""
    print("\n\nBF16 Reference Pattern Analysis")
    print("=" * 70)
    
    # BF16 uses buffer_load...lds with m0 offsets
    # The m0 offset formula encodes both the destination LDS address
    # and implicitly handles bank conflict avoidance
    
    # BF16 swizzle for wave 0: base = 0x8200 = 33280
    # Per-lane offset: (lane&1)*0x80 + (lane>>1)*0x408 + (lane>>5)*16
    
    print("BF16 m0 offset formula: base + (lane&1)*0x80 + (lane>>1)*0x408 + (lane>>5)*16")
    print()
    
    base = 0x8200
    for lane in range(16):
        offset = base + (lane & 1) * 0x80 + (lane >> 1) * 0x408 + (lane >> 5) * 16
        bank = (offset // 4) % 32
        print(f"  Lane {lane:2d}: offset=0x{offset:04x} ({offset:5d}), bank={bank:2d}")
    
    # Check uniqueness
    offsets = [base + (lane & 1) * 0x80 + (lane >> 1) * 0x408 + (lane >> 5) * 16 for lane in range(16)]
    banks = [(o // 4) % 32 for o in offsets]
    print(f"\nUnique banks: {len(set(banks))} (need 16 for zero conflicts)")
    print(f"Banks: {sorted(set(banks))}")

def propose_fp8_swizzle():
    """Propose a swizzle pattern specifically for FP8 MFMA layout"""
    print("\n\nProposed FP8-Specific Swizzle")
    print("=" * 70)
    
    rows = [mfma_row_mapping(lane) for lane in range(16)]
    
    # The issue: rows are [0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23] for lanes 0-15
    # With HD=128 stride, row R has base addr R*128
    # Banks are (R*128/4) % 32 = (R*32) % 32 = 0 for all R!
    
    print("Problem: With HD=128 stride, all rows hit bank 0!")
    print(f"  Rows: {rows}")
    print(f"  Addresses: {[r*128 for r in rows]}")
    print(f"  Banks (no swizzle): {[(r*128//4)%32 for r in rows]}")
    
    # Solution: Use a different row stride in LDS that spreads banks
    # Instead of row*128, use row*stride where stride ensures bank spreading
    
    print("\nSolution: Use LDS stride that spreads banks")
    for stride in [132, 136, 140, 144, 160, 192]:
        addrs = [r * stride for r in rows]
        banks = [(a // 4) % 32 for a in addrs]
        unique = len(set(banks))
        print(f"  Stride {stride}: {unique} unique banks, conflicts={16-unique}")
        if unique == 16:
            print(f"    -> ZERO CONFLICTS! Banks: {banks}")

if __name__ == "__main__":
    analyze_phase_access()
    find_optimal_swizzle()
    analyze_bf16_pattern()
    propose_fp8_swizzle()

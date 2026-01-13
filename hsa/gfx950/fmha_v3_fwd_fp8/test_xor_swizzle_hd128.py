#!/usr/bin/env python3
"""
Evaluate if XOR swizzle is feasible for full HD=128 QK MFMA.

Key questions:
1. Does XOR swizzle work consistently across 8 MFMA iterations?
2. Are there any address conflicts when spanning 4KB (Q) + 4KB (K)?
3. Does the read pattern match the write pattern for all K offsets?
"""

import numpy as np

def xor_swizzle(offset):
    """XOR swizzle formula: offset ^= ((offset & 0x1ff) >> 7) << 3"""
    return offset ^ (((offset & 0x1ff) >> 7) << 3)

def mfma_row_mapping(lane):
    """MFMA A operand row mapping for 32x32x16"""
    row16 = (lane & 3) + ((lane >> 3) & 3) * 4
    row_hi = (lane >> 2) & 1
    return row16 + row_hi * 16

def analyze_swizzle_pattern():
    print("XOR Swizzle Analysis for HD=128 QK MFMA")
    print("=" * 70)
    
    # Q/K matrices are 32 rows × 128 cols (HD=128)
    # Row-major layout: element [row, col] at offset row * 128 + col
    
    print("\n1. Swizzle pattern within 512B window:")
    print("-" * 50)
    # Check first 64 addresses (one wave's worth of 8-byte reads)
    for addr in [0, 8, 16, 24, 32, 64, 128, 256, 384, 504]:
        swiz = xor_swizzle(addr)
        print(f"  addr {addr:3d} -> swizzle {swiz:3d} (diff: {swiz - addr:+3d})")
    
    print("\n2. Swizzle consistency across 512B windows:")
    print("-" * 50)
    # Check if pattern repeats every 512 bytes
    for base in [0, 512, 1024, 2048, 4096]:
        addrs = [base + i for i in [0, 8, 128, 256]]
        swiz = [xor_swizzle(a) for a in addrs]
        print(f"  Base {base:4d}: {addrs} -> {swiz}")
    
    print("\n3. Write-Read consistency for QK MFMA:")
    print("-" * 50)
    
    # For each of 8 MFMA iterations (K=0..15, 16..31, ..., 112..127)
    all_consistent = True
    
    for mfma_iter in range(8):
        k_start = mfma_iter * 16
        
        # Simulate what lanes 0-63 would write and read
        write_addrs = set()
        read_addrs = set()
        
        for lane in range(64):
            row = mfma_row_mapping(lane)
            k_base = 0 if lane < 32 else 8
            
            # Write: lanes 0-31 write row data
            if lane < 32:
                # Write 16 bytes for this row starting at k_start
                for k in range(16):
                    raw_addr = row * 128 + k_start + k  # row-major
                    swiz_addr = xor_swizzle(raw_addr)
                    write_addrs.add((raw_addr, swiz_addr))
            
            # Read: all 64 lanes read 8 bytes
            for k in range(8):
                raw_addr = row * 128 + k_start + k_base + k
                swiz_addr = xor_swizzle(raw_addr)
                read_addrs.add((raw_addr, swiz_addr))
        
        # Check if all read addresses were written
        read_raw = {r[0] for r in read_addrs}
        write_raw = {w[0] for w in write_addrs}
        
        if read_raw <= write_raw:
            status = "✓"
        else:
            status = "✗"
            all_consistent = False
            missing = read_raw - write_raw
            print(f"  MFMA {mfma_iter} (K={k_start:3d}-{k_start+15:3d}): {status} Missing: {missing}")
            continue
        
        # Check swizzle consistency
        read_swiz = {r[1] for r in read_addrs}
        write_swiz = {w[1] for w in write_addrs}
        
        if read_swiz <= write_swiz:
            print(f"  MFMA {mfma_iter} (K={k_start:3d}-{k_start+15:3d}): {status} Swizzle consistent")
        else:
            print(f"  MFMA {mfma_iter} (K={k_start:3d}-{k_start+15:3d}): ✗ Swizzle mismatch!")
            all_consistent = False
    
    print("\n4. Bank conflict analysis (32 banks, 4 bytes each):")
    print("-" * 50)
    
    # For ds_read_b64, check if lanes hit different banks
    for mfma_iter in [0, 4, 7]:  # Sample iterations
        k_start = mfma_iter * 16
        
        # Check bank distribution for first wave (lanes 0-31) 
        banks_phase0 = []  # ds_read_b64 phase 0 (lanes 0-15)
        banks_phase1 = []  # ds_read_b64 phase 1 (lanes 16-31)
        
        for lane in range(32):
            row = mfma_row_mapping(lane)
            k_base = 0  # lanes 0-31 read k=0..7
            raw_addr = row * 128 + k_start + k_base
            swiz_addr = xor_swizzle(raw_addr)
            bank = (swiz_addr // 4) % 32
            
            if lane < 16:
                banks_phase0.append(bank)
            else:
                banks_phase1.append(bank)
        
        # Count bank conflicts
        unique_phase0 = len(set(banks_phase0))
        unique_phase1 = len(set(banks_phase1))
        
        conflicts0 = 16 - unique_phase0
        conflicts1 = 16 - unique_phase1
        
        print(f"  MFMA {mfma_iter}: Phase0 conflicts={conflicts0}, Phase1 conflicts={conflicts1}")
    
    print("\n" + "=" * 70)
    print(f"CONCLUSION: XOR swizzle is {'FEASIBLE' if all_consistent else 'NOT FEASIBLE'}")
    print("=" * 70)
    
    return all_consistent

def compare_with_without_swizzle():
    """Compare bank conflicts with and without swizzle"""
    print("\n\nBank Conflict Comparison: With vs Without XOR Swizzle")
    print("=" * 70)
    
    for mfma_iter in range(8):
        k_start = mfma_iter * 16
        
        banks_no_swiz = []
        banks_swiz = []
        
        for lane in range(16):  # Phase 0 of ds_read_b64
            row = mfma_row_mapping(lane)
            raw_addr = row * 128 + k_start
            swiz_addr = xor_swizzle(raw_addr)
            
            banks_no_swiz.append((raw_addr // 4) % 32)
            banks_swiz.append((swiz_addr // 4) % 32)
        
        conflicts_no = 16 - len(set(banks_no_swiz))
        conflicts_yes = 16 - len(set(banks_swiz))
        
        improvement = conflicts_no - conflicts_yes
        print(f"  K={k_start:3d}-{k_start+15:3d}: No swizzle={conflicts_no} conflicts, "
              f"With swizzle={conflicts_yes} conflicts (improvement: {improvement})")

if __name__ == "__main__":
    feasible = analyze_swizzle_pattern()
    compare_with_without_swizzle()

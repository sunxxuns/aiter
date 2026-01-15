#!/usr/bin/env python3
"""Analyze bank conflicts for FP8 ds_read_b64."""

def get_bank(byte_offset):
    """Get LDS bank number for a byte offset (64 banks, 4 bytes each)."""
    return (byte_offset >> 2) & 63

def analyze_simple_rowmajor():
    """Analyze bank conflicts with simple row-major layout."""
    print("=" * 70)
    print("Simple Row-Major Layout Analysis")
    print("Layout: Q[r,c] at LDS[r*128 + c]")
    print("Thread tid reads Q[tid%32, (tid//32)*8 : (tid//32)*8+8]")
    print("=" * 70)
    
    # ds_read_b64 reads 8 bytes = 2 dwords from 2 consecutive banks
    banks_used = {}  # bank -> list of threads
    
    for tid in range(64):
        row = tid % 32
        k_base = (tid // 32) * 8
        offset = row * 128 + k_base
        
        # ds_read_b64 accesses 2 consecutive 4-byte chunks
        bank0 = get_bank(offset)
        bank1 = get_bank(offset + 4)
        
        for bank in [bank0, bank1]:
            if bank not in banks_used:
                banks_used[bank] = []
            banks_used[bank].append(tid)
    
    print(f"\nBanks used: {len(banks_used)}")
    
    max_conflicts = max(len(tids) for tids in banks_used.values())
    print(f"Max threads per bank: {max_conflicts}")
    print(f"Bank conflict factor: {max_conflicts}x")
    
    if max_conflicts > 1:
        print("\nConflicting banks:")
        for bank, tids in sorted(banks_used.items()):
            if len(tids) > 1:
                print(f"  Bank {bank}: threads {tids}")

def analyze_bf16_style_swizzle():
    """Analyze BF16-style swizzle pattern for FP8."""
    print("\n" + "=" * 70)
    print("BF16-Style Swizzle Analysis (adapted for FP8)")
    print("=" * 70)
    
    # BF16 v2 formula: 0x8200 + (tid>>5)*16 + (lane&1)*128 + (lane>>1)*1032
    # For FP8, we need to adapt this for 8-byte reads instead of 4-byte
    
    banks_used = {}
    
    for tid in range(64):
        lane = tid & 31
        # BF16 formula (byte addresses)
        v2 = 0x8200 + (tid >> 5) * 16 + (lane & 1) * 128 + (lane >> 1) * 1032
        
        # For FP8, same formula but reads 8 bytes
        bank0 = get_bank(v2)
        bank1 = get_bank(v2 + 4)
        
        for bank in [bank0, bank1]:
            if bank not in banks_used:
                banks_used[bank] = []
            banks_used[bank].append(tid)
        
        if tid < 8:
            print(f"tid {tid}: v2=0x{v2:x}, banks={bank0},{bank1}")
    
    print(f"\nBanks used: {len(banks_used)}")
    max_conflicts = max(len(tids) for tids in banks_used.values())
    print(f"Max threads per bank: {max_conflicts}")
    print(f"Bank conflict factor: {max_conflicts}x")

def propose_fp8_swizzle():
    """Propose a swizzle pattern for FP8 that avoids bank conflicts."""
    print("\n" + "=" * 70)
    print("Proposed FP8 Swizzle (bank-conflict-free)")
    print("=" * 70)
    
    # Goal: 64 threads, each reads 8 bytes via ds_read_b64
    # Each read touches 2 consecutive banks
    # Need all 64 threads to hit different bank pairs
    # 
    # With 64 banks and 2 banks per read, max 32 concurrent reads without conflict
    # But we have 64 threads, so some overlap is inevitable
    # 
    # Key insight: within a wavefront (64 threads), reads are serialized if 
    # they hit the same bank. We want each bank touched by at most 1 thread.
    # 
    # Ideal: thread tid accesses banks (tid*2) and (tid*2+1)
    # This requires offset = tid * 8, but we need MFMA-compatible layout
    
    # Alternative: use row XOR swizzle
    # offset = row * 128 + k_base XOR (some function of row)
    
    banks_used = {}
    
    print("\nTrying XOR swizzle: offset = (row * 128 + k_base) XOR (row * 4)")
    for tid in range(64):
        row = tid % 32
        k_base = (tid // 32) * 8
        
        # Simple XOR swizzle
        base_offset = row * 128 + k_base
        swizzle = (row & 0xF) * 8  # Spread based on low bits of row
        offset = base_offset ^ swizzle
        
        bank0 = get_bank(offset)
        bank1 = get_bank(offset + 4)
        
        for bank in [bank0, bank1]:
            if bank not in banks_used:
                banks_used[bank] = []
            banks_used[bank].append(tid)
        
        if tid < 8:
            print(f"tid {tid}: row={row}, k_base={k_base}, offset=0x{offset:x}, banks={bank0},{bank1}")
    
    max_conflicts = max(len(tids) for tids in banks_used.values())
    print(f"\nMax threads per bank: {max_conflicts}")
    print(f"Bank conflict factor: {max_conflicts}x")
    
    # Try another swizzle
    print("\n" + "-" * 40)
    print("Trying stride-132 layout:")
    banks_used = {}
    
    for tid in range(64):
        row = tid % 32
        k_base = (tid // 32) * 8
        
        # Stride 132 (128 + 4) to shift banks between rows
        offset = row * 132 + k_base
        
        bank0 = get_bank(offset)
        bank1 = get_bank(offset + 4)
        
        for bank in [bank0, bank1]:
            if bank not in banks_used:
                banks_used[bank] = []
            banks_used[bank].append(tid)
        
        if tid < 8:
            print(f"tid {tid}: row={row}, k_base={k_base}, offset=0x{offset:x}, banks={bank0},{bank1}")
    
    max_conflicts = max(len(tids) for tids in banks_used.values())
    print(f"\nMax threads per bank: {max_conflicts}")
    print(f"Bank conflict factor: {max_conflicts}x")

if __name__ == "__main__":
    analyze_simple_rowmajor()
    analyze_bf16_style_swizzle()
    propose_fp8_swizzle()

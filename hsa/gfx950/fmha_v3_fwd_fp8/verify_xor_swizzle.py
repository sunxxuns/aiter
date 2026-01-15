#!/usr/bin/env python3
"""
Verify the XOR swizzle pattern and show implementation details.
"""

print("=" * 70)
print("XOR SWIZZLE VERIFICATION: (row * 144 + k_off) XOR (row * 4)")
print("=" * 70)

# Show the address mapping for each lane
print("\nLane-to-Address-to-Bank mapping:")
print(f"{'Lane':>4} | {'Row':>3} | {'k_off':>5} | {'Base':>6} | {'XOR':>5} | {'Addr':>6} | {'Bank':>4}")
print("-" * 55)

banks_hit = set()
bank_map = {}

for lane in range(64):
    row = lane % 32
    k_off = 8 if lane >= 32 else 0
    
    base = row * 144 + k_off
    xor_val = row * 4
    addr = base ^ xor_val
    bank = (addr // 4) % 64
    
    banks_hit.add(bank)
    bank_map[bank] = lane
    
    print(f"{lane:>4} | {row:>3} | {k_off:>5} | {base:>6} | {xor_val:>5} | {addr:>6} | {bank:>4}")

print(f"\nUnique banks hit: {len(banks_hit)}/64")
print(f"Zero conflicts: {len(banks_hit) == 64}")

# Show ASM implementation
print("\n" + "=" * 70)
print("ASM IMPLEMENTATION")
print("=" * 70)

print("""
; Calculate LDS read address with XOR swizzle
; Input: v2 = row (0-31), v3 = k_off (0 or 8)
; Output: v4 = swizzled address

; Method 1: Direct calculation
    v_mul_lo_u32 v4, v2, 144          ; base = row * 144
    v_add_u32 v4, v4, v3              ; base += k_off
    v_lshlrev_b32 v5, 2, v2           ; xor_val = row * 4
    v_xor_b32 v4, v4, v5              ; addr = base XOR xor_val

; Or with immediate (if 144 fits):
; Note: 144 = 0x90, fits in inline constant
    s_mov_b32 s10, 144
    v_mul_lo_u32 v4, v2, s10          ; v4 = row * 144
    v_add_u32 v4, v4, v3              ; v4 += k_off
    v_lshlrev_b32 v5, 2, v2           ; v5 = row << 2 = row * 4  
    v_xor_b32 v4, v4, v5              ; v4 ^= v5

; For ds_read_b64:
    ds_read_b64 v[10:11], v4          ; Read 8 bytes from swizzled address
""")

# Show LDS write pattern (for storing data with same swizzle)
print("\n" + "=" * 70)
print("LDS WRITE PATTERN")
print("=" * 70)

print("""
When writing to LDS, must use same swizzle:

; Write row 'row' of data to LDS with XOR swizzle
; Input: v2 = row, v3 = k_col (0-127 for head_dim), v[10:11] = data
; 
; For FP8, each row is 128 bytes, but we pad to 144 bytes
; Write address = (row * 144 + k_col) XOR (row * 4)

; But typically we write sequentially and read with swizzle
; Or: write with inverse swizzle, read sequentially

; Simpler: write row-major with pitch-144, apply XOR only on read
; Write: addr = row * 144 + k_col (no XOR)
; Read:  addr = (row * 144 + k_off) XOR (row * 4)

; This works if XOR only affects bank selection, not data layout
; Let's verify...
""")

# Verify that XOR doesn't change the data being read
print("Verification that XOR preserves data:")
print("For row 0: base=0, XOR with 0 → addr=0 (same)")
print("For row 1: base=144, XOR with 4 → addr=148")
print("  → reads from byte 148, not byte 144")
print("\nPROBLEM: XOR changes the actual address, not just bank!")
print("We'd be reading wrong data!")

print("\n" + "=" * 70)
print("SOLUTION: Store data with XOR swizzle too")
print("=" * 70)
print("""
When loading Q/K from global to LDS:
1. Thread (row, k) loads from global[row][k]
2. Stores to LDS at address = (row * 144 + k) XOR (row * 4)

When reading for MFMA:
3. Thread reads from same swizzled address

This ensures reads get correct data with zero bank conflicts.
""")

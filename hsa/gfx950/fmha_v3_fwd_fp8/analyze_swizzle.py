#!/usr/bin/env python3
"""
Analyze different swizzle patterns to find zero-conflict solution.
"""

def analyze_pattern(name, addr_func):
    """
    Analyze a given address function for bank conflicts.
    addr_func takes (row, k_offset) and returns byte address.
    """
    banks = {}
    
    for lane in range(64):
        row = lane % 32
        k_off = 8 if lane >= 32 else 0
        addr = addr_func(row, k_off)
        bank = (addr // 4) % 64
        
        if bank not in banks:
            banks[bank] = []
        banks[bank].append((lane, row, k_off))
    
    max_conflict = max(len(v) for v in banks.values())
    num_banks = len(banks)
    
    return num_banks, max_conflict, banks

# Pattern 1: Simple pitch-136 (current)
def pitch136(row, k_off):
    return row * 136 + k_off

# Pattern 2: XOR swizzle
# addr = row * pitch + k_off ^ (row * factor)
def xor_swizzle(row, k_off, factor=4):
    base = row * 136 + k_off
    return base ^ (row * factor)

# Pattern 3: BF16-style swizzle
# Looking at BF16 reference, it uses complex offsets
def bf16_style(row, k_off):
    # BF16 uses stride 0x408 (1032) between waves
    # Within a wave, it uses offsets like 0, 512, 64, 576, ...
    # This creates an interleaved pattern
    wave_in_64 = row // 16
    row_in_16 = row % 16
    # Approximate the BF16 pattern
    return row_in_16 * 136 + wave_in_64 * 512 + k_off

# Pattern 4: Bank-cyclic
# Explicitly assign each row to a different bank
def bank_cyclic(row, k_off):
    # Want row 0 -> bank 0, row 1 -> bank 2, row 2 -> bank 4, ...
    # With k_off, row 0+k8 -> bank 1, row 1+k8 -> bank 3, ...
    # So banks 0,1,2,3,...,63 should be hit exactly once
    
    # For lane L: row = L%32, k_off = 8 if L>=32 else 0
    # Want: bank(L) = L
    # bank = addr / 4 % 64
    # addr = bank * 4 + k_off (but k_off is 0 or 8)
    # 
    # For k_off=0: addr should give bank = row (lanes 0-31)
    # For k_off=8: addr should give bank = row+32 (lanes 32-63)
    
    # If we set addr = row * 256 + k_off * 4:
    # k_off=0: bank = (row*256)/4 % 64 = 64*row % 64 = 0 (bad!)
    
    # Need a different approach
    # addr = row * stride + k_off where stride gives unique banks
    # bank(lane L) = ((L%32)*stride/4 + (L>=32)*2) % 64
    # We want this = L
    
    # For lanes 0-31: bank = (row * stride / 4) % 64 = row
    #   -> stride/4 mod 64 = 1, so stride = 4 (bad, no room for data)
    #   or stride/4 mod 64 = 33, so stride = 132 (might work!)
    
    # Let's try stride = 132
    return row * 132 + k_off

# Pattern 5: Diagonal offset
# Triton's approach: add diagonal padding
def diagonal_pad(row, k_off):
    # pad_offset = row % pad_period
    pad_period = 8  # bytes
    pad = (row * 4) % pad_period  # 0, 4, 0, 4, ...
    return row * (128 + pad_period) + k_off + pad

print("=" * 70)
print("SWIZZLE PATTERN ANALYSIS")
print("=" * 70)

patterns = [
    ("Pitch-136 (current)", pitch136),
    ("XOR swizzle (factor=4)", lambda r, k: xor_swizzle(r, k, 4)),
    ("XOR swizzle (factor=8)", lambda r, k: xor_swizzle(r, k, 8)),
    ("XOR swizzle (factor=16)", lambda r, k: xor_swizzle(r, k, 16)),
    ("Pitch-132", lambda r, k: r * 132 + k),
    ("Diagonal pad", diagonal_pad),
]

for name, func in patterns:
    num_banks, max_conf, banks = analyze_pattern(name, func)
    status = "PERFECT" if max_conf == 1 else f"{max_conf}-way conflict"
    print(f"\n{name}:")
    print(f"  Banks used: {num_banks}/64")
    print(f"  Max conflict: {status}")
    
    if max_conf > 1:
        # Show first few conflicts
        conflicts = [(b, lanes) for b, lanes in banks.items() if len(lanes) > 1]
        for b, lanes in conflicts[:3]:
            print(f"  Bank {b}: lanes {[l[0] for l in lanes]}")

# Now let's find a working XOR factor
print("\n" + "=" * 70)
print("SEARCHING FOR ZERO-CONFLICT XOR FACTOR")
print("=" * 70)

for factor in range(1, 256):
    func = lambda r, k, f=factor: (r * 136 + k) ^ (r * f)
    num_banks, max_conf, _ = analyze_pattern(f"XOR factor={factor}", func)
    if max_conf == 1:
        print(f"  Factor {factor}: ZERO CONFLICT!")
        # Verify this is implementable
        print(f"    Address formula: (row * 136 + k_off) XOR (row * {factor})")
        break
else:
    print("  No simple XOR factor found")

# Try different base strides with XOR
print("\n" + "=" * 70)
print("SEARCHING FOR ZERO-CONFLICT STRIDE + XOR")
print("=" * 70)

for stride in range(128, 160, 4):
    for factor in range(1, 64):
        func = lambda r, k, s=stride, f=factor: (r * s + k) ^ (r * f)
        num_banks, max_conf, _ = analyze_pattern(f"stride={stride}, XOR={factor}", func)
        if max_conf == 1:
            print(f"  Stride {stride}, XOR factor {factor}: ZERO CONFLICT!")
            print(f"    Formula: (row * {stride} + k_off) XOR (row * {factor})")
            break
    else:
        continue
    break
else:
    print("  No stride+XOR combination found in range")

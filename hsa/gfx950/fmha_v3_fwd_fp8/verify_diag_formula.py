#!/usr/bin/env python3
"""
Verify the diagonal padding formula actually eliminates bank conflicts.
"""

def check_formula(name, addr_func):
    """Check bank conflicts for a given address formula."""
    banks = {}
    for lane in range(64):
        row = lane % 32
        k_off = 8 if lane >= 32 else 0
        addr = addr_func(row, k_off)
        bank = (addr // 4) % 64
        
        if bank not in banks:
            banks[bank] = []
        banks[bank].append((lane, row, k_off, addr))
    
    max_conflict = max(len(v) for v in banks.values())
    return max_conflict, banks

# Formula 1: Simple pitch-136 (current baseline)
def pitch136(row, k_off):
    return row * 136 + k_off

# Formula 2: Pitch-136 + diagonal (row % 2) * 4
def pitch136_diag(row, k_off):
    return row * 136 + k_off + (row % 2) * 4

# Formula 3: Various other options
def pitch144_xor(row, k_off):
    return (row * 144 + k_off) ^ (row * 4)

print("=" * 70)
print("BANK CONFLICT VERIFICATION")
print("=" * 70)

formulas = [
    ("Pitch-136 (baseline)", pitch136),
    ("Pitch-136 + (row%2)*4", pitch136_diag),
    ("Pitch-144 XOR (row*4)", pitch144_xor),
]

for name, func in formulas:
    max_conf, banks = check_formula(name, func)
    status = "ZERO" if max_conf == 1 else f"{max_conf}-way"
    print(f"\n{name}:")
    print(f"  Max conflicts: {status}")
    
    if max_conf > 1:
        # Show conflicts
        conflicts = [(b, lanes) for b, lanes in banks.items() if len(lanes) > 1]
        print(f"  Conflicting banks: {len(conflicts)}")
        for b, lanes in conflicts[:3]:
            print(f"    Bank {b}: lanes {[l[0] for l in lanes]}")

# The original analysis must have been wrong. Let me search more
print("\n" + "=" * 70)
print("SEARCHING FOR ACTUAL ZERO-CONFLICT FORMULA WITH PITCH-136")
print("=" * 70)

for period in [2, 4, 8, 16]:
    for shift in range(1, 256):
        func = lambda r, k, p=period, s=shift: r * 136 + k + (r % p) * s
        max_conf, _ = check_formula("test", func)
        if max_conf == 1:
            print(f"FOUND: (row % {period}) * {shift}")
            print(f"  Formula: row * 136 + k_off + (row % {period}) * {shift}")
            break
    else:
        continue
    break
else:
    print("No zero-conflict formula found with pitch-136!")
    
    # Try pitch-144
    print("\nSearching with pitch-144...")
    for period in [2, 4, 8, 16]:
        for shift in range(1, 256):
            func = lambda r, k, p=period, s=shift: r * 144 + k + (r % p) * s
            max_conf, _ = check_formula("test", func)
            if max_conf == 1:
                print(f"FOUND: pitch-144 + (row % {period}) * {shift}")
                break
        else:
            continue
        break
    else:
        print("No simple formula found!")

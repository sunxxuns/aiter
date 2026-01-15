#!/usr/bin/env python3
"""
Find optimal pitch for zero bank conflicts.
"""

def analyze_pitch(pitch):
    """
    For MFMA read pattern:
    - Lane L reads row (L % 32), k_offset = 8 * (L >= 32)
    - Address = row * pitch + k_offset
    - Bank = (addr / 4) % 64
    """
    banks = set()
    bank_map = {}
    
    for lane in range(64):
        row = lane % 32
        k_off = 8 if lane >= 32 else 0
        addr = row * pitch + k_off
        bank = (addr // 4) % 64
        
        if bank in bank_map:
            bank_map[bank].append(lane)
        else:
            bank_map[bank] = [lane]
        banks.add(bank)
    
    max_conflict = max(len(v) for v in bank_map.values())
    return len(banks), max_conflict, bank_map

print("Searching for zero-conflict pitch values...")
print(f"{'Pitch':>6} | {'Banks':>6} | {'MaxConf':>8} | Status")
print("-" * 40)

zero_conflict_pitches = []
for pitch in range(128, 260, 4):  # Must be multiple of 4 for alignment
    num_banks, max_conflict, _ = analyze_pitch(pitch)
    
    if max_conflict == 1:
        status = "*** ZERO CONFLICT ***"
        zero_conflict_pitches.append(pitch)
    elif max_conflict == 2:
        status = "2-way"
    else:
        status = f"{max_conflict}-way"
    
    if max_conflict <= 2 or pitch in [128, 256]:
        print(f"{pitch:>6} | {num_banks:>6} | {max_conflict:>8} | {status}")

print("\n" + "=" * 50)
print("ZERO CONFLICT PITCHES:", zero_conflict_pitches)

if zero_conflict_pitches:
    best_pitch = min(zero_conflict_pitches)
    print(f"\nBest choice: {best_pitch} (minimum padding)")
    print(f"  Overhead: {best_pitch - 128} bytes/row = {(best_pitch-128)/128*100:.1f}%")
    
    # Show bank mapping for best pitch
    _, _, bank_map = analyze_pitch(best_pitch)
    print(f"\nBank mapping for pitch={best_pitch}:")
    for bank in sorted(bank_map.keys())[:8]:
        print(f"  Bank {bank:2d}: lane {bank_map[bank][0]}")
    print("  ...")

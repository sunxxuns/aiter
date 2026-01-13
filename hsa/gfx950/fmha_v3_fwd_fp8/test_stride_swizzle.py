#!/usr/bin/env python3
"""
Test stride-based swizzle (132 or 140) vs XOR swizzle for FP8 QK MFMA.
Verify it works for all 8 MFMA iterations (HD=128).
"""

def mfma_row_mapping(lane):
    row16 = (lane & 3) + ((lane >> 3) & 3) * 4
    row_hi = (lane >> 2) & 1
    return row16 + row_hi * 16

def analyze_stride_for_all_k():
    """Check bank conflicts for all K iterations with stride 132"""
    print("Stride 132 Analysis for All K Iterations")
    print("=" * 70)
    
    stride = 132  # Key: not a multiple of 128
    
    for k_iter in range(8):
        k_start = k_iter * 16
        
        # Phase 0: lanes 0-15
        conflicts_p0 = check_phase_conflicts(0, 16, stride, k_start)
        # Phase 1: lanes 16-31  
        conflicts_p1 = check_phase_conflicts(16, 32, stride, k_start)
        # Phase 2: lanes 32-47
        conflicts_p2 = check_phase_conflicts(32, 48, stride, k_start)
        # Phase 3: lanes 48-63
        conflicts_p3 = check_phase_conflicts(48, 64, stride, k_start)
        
        total = conflicts_p0 + conflicts_p1 + conflicts_p2 + conflicts_p3
        print(f"K={k_start:3d}-{k_start+15:3d}: P0={conflicts_p0}, P1={conflicts_p1}, P2={conflicts_p2}, P3={conflicts_p3}, Total={total}")

def check_phase_conflicts(lane_start, lane_end, stride, k_start):
    """Check bank conflicts for a phase"""
    from collections import defaultdict
    
    bank_count = defaultdict(int)
    
    for lane in range(lane_start, lane_end):
        row = mfma_row_mapping(lane)
        k_base = 0 if lane < 32 else 8
        
        # Address with stride-based layout
        addr = row * stride + k_start + k_base
        bank = (addr // 4) % 32
        bank_count[bank] += 1
    
    conflicts = sum(c - 1 for c in bank_count.values() if c > 1)
    return conflicts

def compare_layouts():
    """Compare stride 128 vs 132 for K=0"""
    print("\n\nComparison: Stride 128 vs 132 for K=0")
    print("=" * 70)
    
    for stride in [128, 132]:
        print(f"\nStride {stride}:")
        print("-" * 40)
        
        total_conflicts = 0
        for phase, (lane_start, lane_end) in enumerate([(0,16), (16,32), (32,48), (48,64)]):
            banks = []
            for lane in range(lane_start, lane_end):
                row = mfma_row_mapping(lane)
                k_base = 0 if lane < 32 else 8
                addr = row * stride + k_base
                bank = (addr // 4) % 32
                banks.append(bank)
            
            unique = len(set(banks))
            conflicts = 16 - unique
            total_conflicts += conflicts
            print(f"  Phase {phase} (lanes {lane_start:2d}-{lane_end-1:2d}): {unique:2d} unique banks, {conflicts:2d} conflicts")
        
        print(f"  Total conflicts: {total_conflicts}")

def lds_size_calculation():
    """Calculate LDS size needed for stride 132"""
    print("\n\nLDS Size Calculation")
    print("=" * 70)
    
    stride = 132
    hd = 128
    rows = 32
    
    # Q: 32 rows × HD columns
    # Last element: row 31, col 127
    # Address = 31 * 132 + 127 = 4092 + 127 = 4219
    q_last = (rows - 1) * stride + (hd - 1)
    q_size = q_last + 1
    
    # K: same layout, offset by Q size
    k_offset = q_size
    k_last = k_offset + (rows - 1) * stride + (hd - 1)
    k_size = k_last + 1 - k_offset
    
    total = k_offset + k_size
    
    print(f"Q region: 0 to {q_last} ({q_size} bytes)")
    print(f"K region: {k_offset} to {k_last} ({k_size} bytes)")
    print(f"Total LDS needed: {total} bytes ({total/1024:.1f} KB)")
    print(f"Available LDS: 65536 bytes (64 KB)")
    print(f"Feasible: {'YES' if total <= 65536 else 'NO'}")

if __name__ == "__main__":
    analyze_stride_for_all_k()
    compare_layouts()
    lds_size_calculation()

# FP8 Flash Attention Kernel Testing Lessons

## Test Summary

| Test Case | Max Error | Status |
|-----------|-----------|--------|
| Random normal (σ=0.5) | 0.097 | ✓ PASS |
| Random normal (σ=1.0) | 2.32 | ✗ FAIL |
| Uniform [0,1] | 0.19 | ✓ PASS |
| Uniform [-1,1] | 0.24 | ✓ PASS |
| Sparse (20% non-zero) | 0.34 | ✓ PASS |
| One-hot pattern | 1.82 | ✗ FAIL |
| Sequential pattern | 28.85 | ✗ FAIL |
| Large values (σ=3.0) | 5.88 | ✗ FAIL |
| Small values (σ=0.01) | 0.00 | ✓ PASS |
| All positive | 0.21 | ✓ PASS |

---

## Key Finding: Numerical Instability with Extreme Attention

### When Kernel Works Well
- **Uniform attention** (Q=K=constant): Max error ≈ 0
- **Random data with small variance** (σ ≤ 0.5): Max error < 0.1
- **Small input values**: Max error ≈ 0

### When Kernel Fails
- **Sequential/structured inputs** that produce large S values before softmax
- **Large input values** (σ > 1.0)
- **Patterns where attention becomes very peaked** (P values near 0 or 1)

### Root Cause Analysis

The `fwd_fp8_kloop.s` kernel uses **online softmax** which computes:
```
running_max, running_sum across K-tiles
P = exp((S - max) * scale)
O = (O * correction + P @ V) / final_sum
```

Issues occur when:
1. **S values are very large** (>50 before scaling)
   - Softmax becomes numerically unstable
   - Most P values become 0 (underflow) or 1 (saturation)

2. **Different K-tiles have very different max values**
   - The correction factor `exp(old_max - new_max)` can overflow/underflow

---

## Failure Pattern Matrix

| Q variation | K variation | V variation | Max Error | Status |
|-------------|-------------|-------------|-----------|--------|
| row-vary | row-vary | row-vary | 14.96 | ✗ |
| row-vary | row-vary | col-vary | 12.08 | ✗ |
| row-vary | row-vary | uniform | 4.83 | ✗ |
| uniform | uniform | any | ~0 | ✓ |
| random | random | any | <0.5 | ✓ |

**Conclusion**: When both Q and K have structured variation (esp. row-varying), 
the resulting attention pattern S = Q @ K^T becomes extreme and causes failures.

---

## Recommendations for Production

1. **Input Scaling**: Ensure Q, K, V are normalized (mean=0, std≈0.5)
2. **FP8 Range**: Keep values within FP8's effective range (-3.5 to 3.5)
3. **Validate Attention Patterns**: Avoid inputs that produce S values > 50

---

## Metadata Lesson (Critical!)

### Problem
Isolated MFMA kernel produced garbage/NaN while working kernel was correct.

### Root Cause
The `.amdhsa_accum_offset` and VGPR count in kernel metadata were too low.

### Fix
```asm
// WRONG (causes MFMA output corruption):
.amdhsa_next_free_vgpr 48
.amdhsa_accum_offset 48

// CORRECT:
.amdhsa_next_free_vgpr 148
.amdhsa_accum_offset 148
```

---

## Python Debug Buffer Lesson (Critical!)

When reading uint32 values from a float32 debug buffer:

```python
# WRONG - int() truncates small floats to 0!
b64_lo = int(d[0].item())  # Returns 0 for 0x2e9b391c!

# CORRECT - interpret raw bits:
import struct
def as_u32(f): return struct.unpack('I', struct.pack('f', f))[0]
b64_lo = as_u32(d[0].item())  # Returns 0x2e9b391c correctly
```

---

## Test Methodology

1. **Don't use uniform inputs** - they hide bugs due to symmetry
2. **Use random data** with controlled variance
3. **Test edge cases**: large values, small values, structured patterns
4. **Compare against PyTorch reference** using same FP8-quantized inputs
5. **Check for NaN** - indicates fundamental computation errors
6. **Acceptable error threshold**: <0.5 for FP8 (limited precision)

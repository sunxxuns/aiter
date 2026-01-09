# FP8 Flash Attention V Layout Issue

## Problem
The kernel incorrectly broadcasts V data across output D positions.

## Current Behavior
- Thread 0: all output positions get V[*, D=0..7] → 0.5
- Thread 1: all output positions get V[*, D=8..15] → 2.0  
- Thread 2+: all output positions get V[*, D=16+] → 1.0

## Expected Behavior
Each output position O[Q, D] should be sum_k(P[Q, K] * V[K, D])
- O[*, D=0..7] = 0.5
- O[*, D=8..15] = 2.0
- O[*, D=16+] = 1.0

## Root Cause
The MFMA B operand needs:
- Each lane n (n=0..31) to provide V[K=0..7, D=n] (8 K values at ONE D position)

Current V reading provides:
- Each lane reads 8 consecutive bytes = V[K, D..D+7] (8 D values at ONE K position)

## Fix Required
1. Store V in LDS with K as inner dimension: V[D, K] (transpose)
2. Use ds_read_b64_tr_b8 to gather 8 K values per lane
3. Or restructure V loading to create correct layout

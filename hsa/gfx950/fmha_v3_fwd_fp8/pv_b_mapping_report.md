# PV B-Operand Mapping Report

- Generated: `2026-01-23T05:04:41Z`
- Source CSV: `/sgl-workspace/aiter/hsa/gfx950/fmha_v3_fwd_fp8/p_pack_mapping.csv`

## Coverage
- Valid mappings: `1024`
- Missing mappings: `0`

## Row LSB vs Byte Parity
- `(src_byte & 1) == (row & 1)` matches: `1023/1024`
- Interpretation: row LSB is encoded in byte parity after mix.

## Diagonal Mapping (row == k, 0..31)
Format: `row -> (lane, byte_pos)` where byte_pos = src_reg*4 + src_byte.

- `00 -> (lane 63, pos 31)`
- `01 -> (lane 01, pos 01)`
- `02 -> (lane 02, pos 02)`
- `03 -> (lane 03, pos 03)`
- `04 -> (lane 04, pos 04)`
- `05 -> (lane 05, pos 05)`
- `06 -> (lane 06, pos 06)`
- `07 -> (lane 07, pos 07)`
- `08 -> (lane 08, pos 08)`
- `09 -> (lane 09, pos 09)`
- `10 -> (lane 10, pos 10)`
- `11 -> (lane 11, pos 11)`
- `12 -> (lane 12, pos 12)`
- `13 -> (lane 13, pos 13)`
- `14 -> (lane 14, pos 14)`
- `15 -> (lane 15, pos 15)`
- `16 -> (lane 48, pos 00)`
- `17 -> (lane 49, pos 01)`
- `18 -> (lane 50, pos 02)`
- `19 -> (lane 51, pos 03)`
- `20 -> (lane 52, pos 04)`
- `21 -> (lane 53, pos 05)`
- `22 -> (lane 54, pos 06)`
- `23 -> (lane 55, pos 07)`
- `24 -> (lane 56, pos 08)`
- `25 -> (lane 57, pos 09)`
- `26 -> (lane 58, pos 10)`
- `27 -> (lane 59, pos 11)`
- `28 -> (lane 60, pos 12)`
- `29 -> (lane 61, pos 13)`
- `30 -> (lane 62, pos 14)`
- `31 -> (lane 63, pos 15)`

## Diagonal Lane Distribution
- lane 01: 1
- lane 02: 1
- lane 03: 1
- lane 04: 1
- lane 05: 1
- lane 06: 1
- lane 07: 1
- lane 08: 1
- lane 09: 1
- lane 10: 1
- lane 11: 1
- lane 12: 1
- lane 13: 1
- lane 14: 1
- lane 15: 1
- lane 48: 1
- lane 49: 1
- lane 50: 1
- lane 51: 1
- lane 52: 1
- lane 53: 1
- lane 54: 1
- lane 55: 1
- lane 56: 1
- lane 57: 1
- lane 58: 1
- lane 59: 1
- lane 60: 1
- lane 61: 1
- lane 62: 1
- lane 63: 2

## Per-Row Byte Histogram (rows 0..7)
- row 00: b0:31, b3:1
- row 01: b1:32
- row 02: b2:32
- row 03: b3:32
- row 04: b0:32
- row 05: b1:32
- row 06: b2:32
- row 07: b3:32

## Notes
- Row LSB parity is preserved in byte parity, but identity-P rowid still collapses, implying MFMA B expects a different byte placement.
- Diagonal entries show lane remapping for row 0 and rows 16..31; this likely indicates missing lane/byte permute after mix.

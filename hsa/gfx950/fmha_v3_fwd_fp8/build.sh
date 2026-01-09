#!/bin/bash
# Build script for FP8 Flash Attention kernel
# Target: gfx950 (MI300X/MI350)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
CLANG="${ROCM_PATH}/llvm/bin/clang"
LLD="${ROCM_PATH}/llvm/bin/ld.lld"

TARGET_ARCH="gfx950"

echo "=== Building FP8 Flash Attention Kernel ==="
echo "Target: ${TARGET_ARCH}"
echo ""

# Compile assembly to object
echo "[1/2] Assembling fwd_hd128_fp8.s..."
${CLANG} -x assembler \
    -target amdgcn-amd-amdhsa \
    -mcpu=${TARGET_ARCH} \
    -mno-xnack \
    -c "${SCRIPT_DIR}/fwd_hd128_fp8.s" \
    -o "${SCRIPT_DIR}/fwd_hd128_fp8.o"

# Link to code object
echo "[2/2] Linking to fwd_hd128_fp8.co..."
${LLD} -shared \
    "${SCRIPT_DIR}/fwd_hd128_fp8.o" \
    -o "${SCRIPT_DIR}/fwd_hd128_fp8.co"

# Cleanup
rm -f "${SCRIPT_DIR}/fwd_hd128_fp8.o"

echo ""
echo "=== Build complete ==="
echo "Output: ${SCRIPT_DIR}/fwd_hd128_fp8.co"

# Verify
if [ -f "${SCRIPT_DIR}/fwd_hd128_fp8.co" ]; then
    echo ""
    echo "Kernel info:"
    ${ROCM_PATH}/llvm/bin/llvm-readelf -h "${SCRIPT_DIR}/fwd_hd128_fp8.co" 2>/dev/null | grep -E "Machine|Flags" || true
    echo ""
    echo "Symbols:"
    ${ROCM_PATH}/llvm/bin/llvm-nm "${SCRIPT_DIR}/fwd_hd128_fp8.co" 2>/dev/null || true
fi

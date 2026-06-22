#!/usr/bin/env bash
# Reproducible build + test entry point. Used as the verification step for every phase.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build}"

# Fetch prebuilt deps if missing (pdfium binary is gitignored).
./scripts/fetch_pdfium.sh || echo "WARN: pdfium fetch failed; PDF targets will be skipped"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
ctest --test-dir "$BUILD_DIR" --output-on-failure

#!/usr/bin/env bash
# Build a trimmed, macOS/arm64 *static* libpdfium.a from the pdfium source tree, with no
# gn/depot_tools. Pipeline: fetch external deps -> patch out abseil -> CMake build.
# Output: third_party/pdfium-static/{lib/libpdfium.a, include/...}.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SRC="$ROOT/third_party/.pdfium-src"
DEST="$ROOT/third_party/pdfium-static"
BLD="$ROOT/third_party/.pdfium-cmake-build"

if [ -f "$DEST/lib/libpdfium.a" ]; then
  echo "static pdfium already built in $DEST"; exit 0
fi

# 1. source + external deps (idempotent)
if [ ! -d "$SRC/core" ]; then
  echo "[pdfium] cloning source ..."
  git clone --depth 1 https://pdfium.googlesource.com/pdfium "$SRC"
fi
./scripts/fetch_pdfium_deps.sh
./scripts/patch_pdfium_src.sh

# 2. configure + build the trimmed static lib
echo "[pdfium] configuring CMake ..."
cmake -S "$ROOT/third_party/pdfium-cmake" -B "$BLD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DPDFIUM_SRC="$SRC"
echo "[pdfium] building (large) ..."
cmake --build "$BLD" -j"$(sysctl -n hw.ncpu)"

# 3. install: libpdfium.a + public headers (flat, matching #include "fpdfview.h")
mkdir -p "$DEST/lib" "$DEST/include"
cp -f "$BLD/libpdfium.a" "$DEST/lib/"
cp -R "$SRC/public/"* "$DEST/include/"
echo "[pdfium] done -> $DEST/lib/libpdfium.a"
ls -la "$DEST/lib/libpdfium.a"

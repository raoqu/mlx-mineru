#!/usr/bin/env bash
# Build mlx-mineru. Fetches gitignored deps (pdfium binary, tokenizer files) if
# missing, configures CMake, and builds everything (libs, CLI, tests).
#
# Usage:
#   ./build.sh                 # configure + build (Release)
#   ./build.sh --debug         # Debug build
#   ./build.sh --test          # build, then run ctest
#   ./build.sh --weights       # also fetch the ~2.2GB model weights first
#   ./build.sh --mumodel       # also pre-fetch the ~3.2GB mumodel runtime bundle
#   ./build.sh --clean         # remove build/ first (fresh configure)
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="Release"
RUN_TESTS=0
FETCH_WEIGHTS=0
FETCH_MUMODEL=0

for arg in "$@"; do
  case "$arg" in
    --debug)   BUILD_TYPE="Debug" ;;
    --test)    RUN_TESTS=1 ;;
    --weights) FETCH_WEIGHTS=1 ;;
    --mumodel) FETCH_MUMODEL=1 ;;
    --clean)   rm -rf "$BUILD_DIR" ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- dependencies (gitignored; fetched/built on demand) ---
# pdfium + ONNX Runtime have no upstream static build, so they are bundled as dylibs next
# to the binary (relocatable via @loader_path); MLX and OpenCV are built static from source
# so the binary carries no libmlx/libjaccl/opencv dylib. All steps are idempotent (skip if
# already present), so only the first build pays the one-time source-build cost.
# pdfium: build a trimmed, macOS-only static libpdfium.a from source (no dylib at runtime).
# Falls back to the prebuilt dylib if the from-source build is unavailable.
./scripts/build_pdfium_static_src.sh || { echo "WARN: static pdfium build failed; falling back to the prebuilt dylib"; ./scripts/fetch_pdfium.sh || echo "WARN: pdfium fetch failed; PDF targets will be skipped"; }
./scripts/fetch_onnxruntime.sh   || echo "WARN: ONNX Runtime fetch failed; pipeline backend will be skipped"
./scripts/build_opencv_static.sh || echo "WARN: static OpenCV build failed; wired-table path uses reimplemented cv ops"
./scripts/build_mlx_static.sh    || echo "WARN: static MLX build failed (needs the Metal toolchain: xcodebuild -downloadComponent MetalToolchain); falling back to the dynamic pip mlx dylib"
./scripts/fetch_mlx.sh           || echo "WARN: dynamic MLX vendoring failed (only used if the static build is absent)"
./scripts/fetch_tokenizer.sh     || echo "WARN: tokenizer fetch failed; tokenizer test will fail"
if [ "$FETCH_WEIGHTS" = "1" ]; then
  ./scripts/fetch_weights.sh
fi
if [ "$FETCH_MUMODEL" = "1" ]; then
  ./scripts/fetch_mumodel.sh
fi

# --- web UI: build the frontend (pnpm + vite) and embed it into the binary ---
# Skipped gracefully if pnpm is unavailable (binary still builds; --web serves API only).
if command -v pnpm >/dev/null 2>&1; then
  echo "[build] web UI: pnpm install + vite build ..."
  ( cd web && pnpm install && pnpm build ) || echo "WARN: web build failed; --web will serve API only"
else
  echo "WARN: pnpm not found; skipping web UI build (--web will serve API only)"
fi
python3 ./scripts/embed_web.py || echo "WARN: embed_web.py failed"

# --- configure + build ---
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" -j"$JOBS"

if [ "$RUN_TESTS" = "1" ]; then
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo
echo "Built: $BUILD_DIR/mlx-mineru"
echo "Run:   ./$BUILD_DIR/mlx-mineru -p <file.pdf> -o output"

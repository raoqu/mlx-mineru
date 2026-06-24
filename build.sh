#!/usr/bin/env bash
# Build mlx-mineru. Fetches gitignored deps (pdfium binary, tokenizer files) if
# missing, configures CMake, and builds everything (libs, CLI, tests).
#
# Usage:
#   ./build.sh                 # configure + build (Release)
#   ./build.sh --debug         # Debug build
#   ./build.sh --test          # build, then run ctest
#   ./build.sh --weights       # also fetch the ~2.2GB model weights first
#   ./build.sh --clean         # remove build/ first (fresh configure)
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="Release"
RUN_TESTS=0
FETCH_WEIGHTS=0

for arg in "$@"; do
  case "$arg" in
    --debug)   BUILD_TYPE="Debug" ;;
    --test)    RUN_TESTS=1 ;;
    --weights) FETCH_WEIGHTS=1 ;;
    --clean)   rm -rf "$BUILD_DIR" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- dependencies (gitignored; fetched on demand) ---
./scripts/fetch_pdfium.sh    || echo "WARN: pdfium fetch failed; PDF targets will be skipped"
./scripts/fetch_mlx.sh       || echo "WARN: MLX vendoring failed; build will fall back to the pip mlx path"
./scripts/fetch_tokenizer.sh || echo "WARN: tokenizer fetch failed; tokenizer test will fail"
if [ "$FETCH_WEIGHTS" = "1" ]; then
  ./scripts/fetch_weights.sh
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

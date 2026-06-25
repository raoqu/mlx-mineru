#!/usr/bin/env bash
# Build MLX (C++/Metal) as a *static* library from source into third_party/mlx-static/,
# so mlx-mineru links libmlx statically and no longer loads libmlx.dylib / libjaccl.dylib
# at runtime. The Metal shader archive (mlx.metallib) is NOT a dylib — it is a data asset
# that ships next to the binary and is loaded at runtime.
#
# Pinned to the same MLX version the project was built against (0.31.2) so the vendored
# C++ headers/API match.
set -euo pipefail
cd "$(dirname "$0")/.."

VER="${MLX_VERSION:-0.31.2}"
SRC="third_party/.mlx-src"
BLD="third_party/.mlx-build"
DEST="$(pwd)/third_party/mlx-static"

if [ -f "$DEST/lib/libmlx.a" ]; then
  echo "static MLX already built in $DEST"
  exit 0
fi

if [ ! -d "$SRC/.git" ]; then
  echo "[mlx] cloning v$VER ..."
  rm -rf "$SRC"
  git clone --depth 1 --branch "v$VER" https://github.com/ml-explore/mlx.git "$SRC"
fi

echo "[mlx] configuring static build ..."
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEST" \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_METAL=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64

echo "[mlx] building (Metal kernel compile is slow) ..."
cmake --build "$BLD" -j"$(sysctl -n hw.ncpu)"
cmake --install "$BLD"

echo "[mlx] static MLX installed -> $DEST"
find "$DEST" -name "*.a" -o -name "mlx.metallib" 2>/dev/null | sed 's/^/  /'

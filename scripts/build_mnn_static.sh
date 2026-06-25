#!/usr/bin/env bash
# Build MNN (alibaba/MNN) as a static library from source into third_party/mnn-static/, with
# the CPU + Metal backends and ARMv8.2 (fp16/i8) kernels, for benchmarking / a possible MNN
# inference backend. No converter (we use the pip `mnnconvert` at setup time).
# Output: third_party/mnn-static/{lib/libMNN.a, include/...}
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VER="${MNN_VERSION:-3.6.0}"  # 3.0.0 had SiLU/If runtime bugs; 3.6.0 runs layout + ocr_rec
SRC="$ROOT/third_party/.mnn-src"
BLD="$SRC/build_static"
DEST="$ROOT/third_party/mnn-static"

if [ -f "$DEST/lib/libMNN.a" ]; then
  echo "static MNN already built in $DEST"; exit 0
fi

if [ ! -d "$SRC/.git" ]; then
  echo "[mnn] cloning v$VER ..."
  rm -rf "$SRC"
  git clone --depth 1 --branch "$VER" https://github.com/alibaba/MNN.git "$SRC" \
    || git clone --depth 1 https://github.com/alibaba/MNN.git "$SRC"
fi

echo "[mnn] configuring static build (CPU + Metal + CoreML + ARM82) ..."
# Metal (GPU) and CoreML (ANE/GPU) backends are built in alongside CPU so the runner can pick a
# backend at load time (MINERU_MNN_BACKEND=cpu|metal|coreml|auto) and fall back to CPU on op gaps.
# Metal shaders are embedded as source (AllShader.cpp, compiled at runtime via
# newLibraryWithSource) — no external mnn.metallib, so the binary stays relocatable/dylib-free.
# Both backends add object files into the single libMNN.a (MNN_SEP_BUILD=OFF); the final binary
# must additionally link the Apple frameworks (see CMakeLists.txt MNN_FRAMEWORKS).
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DMNN_BUILD_SHARED_LIBS=OFF \
  -DMNN_SEP_BUILD=OFF \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_METAL=ON \
  -DMNN_COREML=ON \
  -DMNN_ARM82=ON \
  -DMNN_USE_THREAD_POOL=ON

echo "[mnn] building ..."
cmake --build "$BLD" -j"$(sysctl -n hw.ncpu)"

mkdir -p "$DEST/lib" "$DEST/include"
find "$BLD" -name "libMNN.a" -exec cp -f {} "$DEST/lib/" \;
cp -R "$SRC/include/MNN" "$DEST/include/"
echo "[mnn] done -> $DEST"
ls -la "$DEST/lib/"*.a 2>/dev/null

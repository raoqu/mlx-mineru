#!/usr/bin/env bash
# Build ONNX Runtime as a *static* library (CPU provider only) from source, so mlx-mineru
# links it statically and carries no libonnxruntime.dylib at runtime. ORT ships no mac-arm64
# static build, so this is from source (heavy: ~30-60 min, downloads protobuf/abseil/onnx/...).
#
# The project uses only the default CPU EP (Session/Value/Env), so we disable every optional
# provider and feature. All component archives are merged into one libonnxruntime.a; the
# consumer must link it with -Wl,-force_load so ORT's static kernel-registration initializers
# are kept (otherwise inference fails with "no suitable kernel").
#
# Output: third_party/onnxruntime-static/{lib/libonnxruntime.a, include/...}
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VER="${ORT_VERSION:-1.26.0}"
SRC="$ROOT/third_party/.onnxruntime-src"
DEST="$ROOT/third_party/onnxruntime-static"
BLD="$SRC/build_static"

if [ -f "$DEST/lib/libonnxruntime.a" ]; then
  echo "static ONNX Runtime already built in $DEST"; exit 0
fi

if [ ! -d "$SRC/.git" ]; then
  echo "[ort] cloning v$VER (shallow, with submodules) ..."
  rm -rf "$SRC"
  git clone --depth 1 --branch "v$VER" --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "$SRC"
fi

# Pre-download all third-party dep archives into a local mirror (per-file curl retries) so
# the build never depends on FetchContent's single-shot downloads over a flaky network.
MIRROR="$SRC/.deps-mirror"
mkdir -p "$MIRROR"
echo "[ort] mirroring deps from cmake/deps.txt ..."
while IFS=';' read -r name url sha; do
  case "$name" in ''|\#*) continue;; esac
  [ -z "$url" ] && continue
  # ORT rewrites https://host/path -> ${MIRROR}/host/path, so mirror the full URL path.
  out="$MIRROR/${url#https://}"
  [ -s "$out" ] && continue
  mkdir -p "$(dirname "$out")"
  for i in 1 2 3 4 5; do
    curl -fsSL --retry 4 --retry-all-errors -o "$out" "$url" && break || { echo "  retry $i: $name"; rm -f "$out"; sleep 2; }
  done
done < "$SRC/cmake/deps.txt"

echo "[ort] building static CPU-only (slow) ..."
# --cmake_deps_mirror_dir: resolve deps from the local mirror (no network during configure).
# FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER: fetch/build ALL deps instead of find_package()-ing
# incompatible system/anaconda ones (e.g. a partial abseil on the path).
# --no_kleidiai + USE_SVE=OFF: drop ARM SVE/KleidiAI kernels (Apple Silicon has no SVE).
python3 "$SRC/tools/ci_build/build.py" \
  --build_dir "$BLD" \
  --config Release \
  --parallel \
  --skip_tests \
  --skip_submodule_sync \
  --no_kleidiai \
  --cmake_deps_mirror_dir "$MIRROR" \
  --osx_arch arm64 \
  --apple_deploy_target 13.0 \
  --cmake_extra_defines \
      onnxruntime_BUILD_SHARED_LIB=OFF \
      onnxruntime_BUILD_UNIT_TESTS=OFF \
      onnxruntime_USE_SVE=OFF \
      FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
      CMAKE_OSX_ARCHITECTURES=arm64

# re2 is a transitive dep whose static lib isn't built unless a final binary links it; build
# it explicitly so its objects are available to merge.
cmake --build "$BLD/Release" --target re2 -j"$(sysctl -n hw.ncpu)" >/dev/null 2>&1 || true

# Package two archives:
#   libonnxruntime.a          = all component + dep archives EXCEPT providers (linked normally;
#                               the linker de-dupes onnx/onnx_proto/protobuf overlaps).
#   libonnxruntime_providers.a = the CPU-kernel static-registration archive (force_load'd by the
#                               consumer so the kernels actually register at runtime).
echo "[ort] packaging static archives ..."
mkdir -p "$DEST/lib" "$DEST/include"
CORE="$(find "$BLD/Release" -name '*.a' | grep -vE "test|gtest|gmock|benchmark|libonnxruntime_providers\.a")"
libtool -static -o "$DEST/lib/libonnxruntime.a" $CORE 2>/dev/null
cp -f "$BLD/Release/libonnxruntime_providers.a" "$DEST/lib/"

# Public C/C++ API headers.
cp -f "$SRC"/include/onnxruntime/core/session/*.h "$DEST/include/" 2>/dev/null || true
echo "[ort] done -> $DEST/lib/{libonnxruntime.a, libonnxruntime_providers.a}"
ls -la "$DEST/lib/"*.a

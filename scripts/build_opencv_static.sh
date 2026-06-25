#!/usr/bin/env bash
# Build a minimal *static* OpenCV (core + imgproc only) from source into
# third_party/opencv/, so mlx-mineru links it statically and no longer depends on the
# Homebrew opencv dylibs (/opt/homebrew/opt/opencv/...) at runtime.
#
# Pinned to 4.13.0 to match the Homebrew build the project was developed against, so the
# wired-table UNet cv ops (resize / morphology / minAreaRect / connectedComponents) stay
# behaviourally aligned. Only core+imgproc are built; no imgcodecs / dnn / image-format
# libs, so the static archives pull in no third-party dylibs.
set -euo pipefail
cd "$(dirname "$0")/.."

VER="${OPENCV_VERSION:-4.13.0}"
SRC="third_party/.opencv-src"
BLD="third_party/.opencv-build"
DEST="$(pwd)/third_party/opencv"

if [ -f "$DEST/lib/libopencv_core.a" ] && [ -f "$DEST/lib/libopencv_imgproc.a" ]; then
  echo "static OpenCV already built in $DEST"
  exit 0
fi

if [ ! -d "$SRC/.git" ]; then
  echo "[opencv] cloning $VER ..."
  rm -rf "$SRC"
  git clone --depth 1 --branch "$VER" https://github.com/opencv/opencv.git "$SRC"
fi

echo "[opencv] configuring static build (core+imgproc) ..."
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DEST" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_LIST=core,imgproc \
  -DOPENCV_GENERATE_PKGCONFIG=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
  -DBUILD_opencv_gapi=OFF -DWITH_ADE=OFF \
  -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF \
  -DWITH_PROTOBUF=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
  -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF \
  -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF \
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_EIGEN=OFF \
  -DWITH_1394=OFF -DWITH_VTK=OFF -DWITH_QUIRC=OFF \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DCMAKE_OSX_ARCHITECTURES=arm64

echo "[opencv] building ..."
cmake --build "$BLD" -j"$(sysctl -n hw.ncpu)"
cmake --install "$BLD"

echo "[opencv] static OpenCV installed -> $DEST"
ls -la "$DEST/lib/"*.a 2>/dev/null || true

#!/usr/bin/env bash
# Vendor the MLX C++ runtime (libmlx.dylib + mlx.metallib) into third_party/mlx/
# so the built binary loads them from the repo, NOT from a Python site-packages
# tree. libmlx is a pure C++/Metal library — no Python interpreter is involved at
# runtime. The pip `mlx` package is used here only as the *source* of the prebuilt
# library (build/setup time); the resulting binary needs no Python install.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="third_party/mlx"

if [ -f "$DEST/lib/libmlx.dylib" ] && [ -f "$DEST/lib/mlx.metallib" ]; then
  echo "vendored MLX already present in $DEST/lib"
  exit 0
fi

# Locate the pip mlx package (build/setup time only).
MLX_ROOT="$(python3 -c 'import mlx.core,os;print(os.path.dirname(mlx.core.__file__))' 2>/dev/null || true)"
if [ -z "${MLX_ROOT:-}" ] || [ ! -f "$MLX_ROOT/lib/libmlx.dylib" ]; then
  echo "ERROR: could not find the pip 'mlx' package (need it once to vendor libmlx)." >&2
  echo "       pip install mlx, then re-run." >&2
  exit 1
fi

mkdir -p "$DEST/lib"
# All MLX runtime dylibs (libmlx.dylib + its private deps, e.g. libjaccl.dylib)
# and the Metal shader archive.
cp -f "$MLX_ROOT/lib/"*.dylib "$DEST/lib/"
cp -f "$MLX_ROOT/lib/mlx.metallib" "$DEST/lib/"   # Metal shaders, loaded next to libmlx
# Vendor headers too so the build doesn't reach into site-packages either.
rm -rf "$DEST/include"
cp -R "$MLX_ROOT/include" "$DEST/include"

# Ensure the dylib id is @rpath-relative (it already is in the pip build).
install_name_tool -id @rpath/libmlx.dylib "$DEST/lib/libmlx.dylib" 2>/dev/null || true
echo "vendored MLX -> $DEST  (libmlx.dylib $(du -h "$DEST/lib/libmlx.dylib" | cut -f1), mlx.metallib $(du -h "$DEST/lib/mlx.metallib" | cut -f1))"

#!/usr/bin/env bash
# Vendor ONNX Runtime (C++) into third_party/onnxruntime/ for the pipeline backend.
# The runtime dylib is taken from the pip `onnxruntime` package (the GitHub tgz
# tends to drop on flaky networks); the small C++ API headers are fetched from the
# matching version tag. No Python at runtime — only the C++ lib is used.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="third_party/onnxruntime"
mkdir -p "$DEST/lib" "$DEST/include"

ORT_PKG="$(python3 -c 'import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))' 2>/dev/null || true)"
VER="$(python3 -c 'import onnxruntime;print(onnxruntime.__version__)' 2>/dev/null || true)"
if [ -z "${ORT_PKG:-}" ]; then
  echo "ERROR: pip 'onnxruntime' not found (pip install onnxruntime)." >&2; exit 1
fi
DYLIB="$(ls "$ORT_PKG"/capi/libonnxruntime.*.dylib 2>/dev/null | head -1)"
cp -f "$DYLIB" "$DEST/lib/"
( cd "$DEST/lib" && ln -sf "$(basename "$DYLIB")" libonnxruntime.dylib )
install_name_tool -id "@rpath/$(basename "$DYLIB")" "$DEST/lib/$(basename "$DYLIB")" 2>/dev/null || true

BASE="https://raw.githubusercontent.com/microsoft/onnxruntime/v${VER}/include/onnxruntime/core/session"
complete() { tail -3 "$1" 2>/dev/null | grep -qE "#endif|namespace|ORT_RUNTIME_CLASS|^\}"; }
for h in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h \
         onnxruntime_ep_c_api.h onnxruntime_run_options_config_keys.h \
         onnxruntime_session_options_config_keys.h; do
  if [ -s "$DEST/include/$h" ] && complete "$DEST/include/$h"; then continue; fi
  for try in $(seq 1 12); do  # flaky network truncates files; retry until complete
    curl -fsSL -C - --max-time 60 -o "$DEST/include/$h" "$BASE/$h" 2>/dev/null || true
    complete "$DEST/include/$h" && break
    sleep 2
  done
done
echo "onnxruntime $VER vendored -> $DEST"

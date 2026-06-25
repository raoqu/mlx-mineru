#!/usr/bin/env bash
# Pre-fetch the mumodel runtime bundle (VLM safetensors + pipeline ONNX, ~3.2GB) that the
# mlx-mineru binary loads at inference time. Idempotent and resumable.
#
# Uses the vendored getmodel downloader (third_party/getmodel, github.com/raoqu/getmodel):
# REST-API based with automatic HF<->ModelScope source selection, byte-range resume, and
# retry — no git / git-lfs. Builds a small getmodel CLI on first run (needs a C++17 compiler
# + system libcurl), then invokes its DownloadModel entry point.
#
# Usage:
#   ./scripts/fetch_mumodel.sh            # -> ./mumodel
#   ./scripts/fetch_mumodel.sh DEST       # -> DEST
#
# At runtime mlx-mineru also auto-downloads this bundle on first use (same getmodel path), so
# this script is only needed to pre-fetch (CI / offline-prep) or to control the destination.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-mumodel}"
HF_URL="https://huggingface.co/raoqu/mlx-mu"
MS_URL="https://modelscope.cn/models/iwannaido/mlx-mu"
GM_SRC="third_party/getmodel"
GM_BIN="third_party/.getmodel-bin/getmodel"

complete() {
  local f="$DEST/MinerU2.5-tokenizer/model.safetensors"
  [ -f "$f" ] && [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)" -gt 10485760 ]
}

if complete; then
  echo "mumodel already present in $DEST/"
  exit 0
fi

# Build the getmodel CLI from vendored source if needed.
if [ ! -x "$GM_BIN" ]; then
  echo "[mumodel] building getmodel ..."
  mkdir -p "$(dirname "$GM_BIN")"
  c++ -std=c++17 -O2 -o "$GM_BIN" "$GM_SRC/getmodel.cpp" "$GM_SRC/main.cpp" -lcurl
fi

# getmodel auto-selects whichever of HF/ModelScope is reachable, resumes, and retries.
"$GM_BIN" --target "$DEST" --retry 5 "$HF_URL" "$MS_URL"

complete && echo "mumodel ready: $DEST/" || { echo "error: mumodel download incomplete" >&2; exit 1; }

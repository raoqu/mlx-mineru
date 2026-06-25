#!/usr/bin/env bash
# Fetch the mumodel runtime bundle (VLM safetensors + pipeline ONNX, ~3.2GB) that the
# mlx-mineru binary loads at inference time. Idempotent and resumable.
#
# This mirrors the index-tts2-metal model-distribution method: the bundle is a single
# repo cloned into a folder the app auto-discovers (default: ./mumodel next to the repo
# / executable). Hugging Face is primary; ModelScope is the mainland fallback.
#
# Usage:
#   ./scripts/fetch_mumodel.sh            # -> ./mumodel
#   ./scripts/fetch_mumodel.sh DEST       # -> DEST
#
# At runtime mlx-mineru also auto-downloads this bundle on first use if it is missing,
# so this script is only needed to pre-fetch (e.g. in CI or offline-prep) or to control
# the destination.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-mumodel}"
HF_URL="https://huggingface.co/raoqu/mlx-mu"
MS_URL="https://modelscope.cn/models/iwannaido/mlx-mu"

complete() {
  # ~2.2GB VLM weights present (and not an LFS pointer stub) => bundle is usable.
  local f="$DEST/MinerU2.5-tokenizer/model.safetensors"
  [ -f "$f" ] && [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)" -gt 10485760 ]
}

if complete; then
  echo "mumodel already present in $DEST/"
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required (and git-lfs for the large weights)." >&2
  exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
  echo "WARN: git-lfs not found; large weights may download as pointer stubs." >&2
  echo "      Install it first:  brew install git-lfs && git lfs install" >&2
fi

clone() {  # clone <url> <dest>
  rm -rf "$2"
  echo "Cloning $1 -> $2"
  git clone "$1" "$2" || return 1
  complete || git -C "$2" lfs pull || true
  complete
}

if clone "$HF_URL" "$DEST"; then
  echo "mumodel ready (Hugging Face): $DEST/"
  exit 0
fi
echo "Hugging Face failed; trying ModelScope ..." >&2
if clone "$MS_URL" "$DEST"; then
  echo "mumodel ready (ModelScope): $DEST/"
  exit 0
fi

echo "error: could not fetch the mumodel bundle from either source." >&2
exit 1

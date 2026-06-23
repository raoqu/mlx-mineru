#!/usr/bin/env bash
# Remove build artifacts. By default cleans the CMake build dir and CLI output.
#
# Usage:
#   ./clean.sh           # remove build/ and output/
#   ./clean.sh --deps    # also remove fetched pdfium + reference clones
#                        # (NOT the model weights/tokenizer — those are large)
#   ./clean.sh --all     # also remove models/ (weights + tokenizer, ~2.2GB)
set -euo pipefail
cd "$(dirname "$0")"

DEPS=0
ALL=0
for arg in "$@"; do
  case "$arg" in
    --deps) DEPS=1 ;;
    --all)  DEPS=1; ALL=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

rm -rf build output
echo "removed build/ and output/"

if [ "$DEPS" = "1" ]; then
  rm -rf third_party/pdfium third_party/reference
  echo "removed third_party/pdfium and third_party/reference (re-fetch with build.sh / scripts/fetch_reference.sh)"
fi

if [ "$ALL" = "1" ]; then
  rm -rf models/MinerU2.5-tokenizer
  echo "removed models/MinerU2.5-tokenizer (re-fetch with scripts/fetch_weights.sh)"
fi

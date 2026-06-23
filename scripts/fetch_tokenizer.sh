#!/usr/bin/env bash
# Fetch the Qwen2 tokenizer + config files for the MinerU2.5 VLM model into
# models/MinerU2.5-tokenizer/ (gitignored). These are small (~16MB); the full
# model weights are fetched separately (Phase 6).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="models/MinerU2.5-tokenizer"
REPO="opendatalab/MinerU2.5-Pro-2605-1.2B"
mkdir -p "$DEST"

need=(tokenizer_config.json vocab.json merges.txt tokenizer.json config.json
      generation_config.json preprocessor_config.json)
for f in "${need[@]}"; do
  if [ -s "$DEST/$f" ]; then continue; fi
  echo "Fetching $f"
  curl -fsSL -o "$DEST/$f" "https://huggingface.co/$REPO/resolve/main/$f"
done
echo "tokenizer files ready in $DEST/"

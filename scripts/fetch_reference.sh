#!/usr/bin/env bash
# Fetch the upstream reference sources we align against (NOT committed; large +
# own licenses). Per AGENT.md principle "对齐优先": port from these, don't guess.
#   - mineru-vl-utils: VLM client, two-step extract, prompts, output grammar,
#     post-processing (otsl2html, equation fixes, json2markdown). Needed for Phase 4.
#   - mlx-vlm: Qwen2-VL MLX architecture (models/qwen2_vl) to port the VLM model.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="third_party/reference"
mkdir -p "$DEST"

clone() {  # repo_url dir
  if [ -d "$DEST/$2/.git" ]; then
    echo "$2 already present"
  else
    git clone --depth 1 "$1" "$DEST/$2"
  fi
}

clone https://github.com/opendatalab/mineru-vl-utils.git mineru-vl-utils
clone https://github.com/Blaizzy/mlx-vlm.git mlx-vlm
echo "Reference sources in $DEST/"

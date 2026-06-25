#!/usr/bin/env bash
# Generate MNN (.mnn) models for the two pipeline models that MNN runs faster with EXACT
# parity — table-cls (PP-LCNet) and wired-table UNet — placing each `.mnn` next to its
# `.onnx`. When present (and the binary was built with MNN, ./build.sh --mnn), mlx-mineru
# uses the MNN model automatically; otherwise it falls back to ONNX Runtime. No other models
# are converted: MNN can't run det/rec/layout/slanet (op-coverage gaps).
#
# Needs the pip `mnnconvert` (pip install MNN) and the pipeline models present.
# Usage: ./scripts/convert_pipeline_mnn.sh [<pipeline_dir>]   (default: mumodel/pipeline)
set -euo pipefail
cd "$(dirname "$0")/.."

PIPE="${1:-mumodel/pipeline}"
if ! command -v mnnconvert >/dev/null 2>&1; then
  echo "error: mnnconvert not found — pip install MNN" >&2; exit 1
fi

# model.onnx paths that MNN handles with exact parity.
MODELS=(
  "$PIPE/TabCls/PP-LCNet_x1_0_table_cls.onnx"
  "$PIPE/TabRec/UnetStructure/unet.onnx"
)

for onnx in "${MODELS[@]}"; do
  [ -f "$onnx" ] || { echo "skip (missing): $onnx"; continue; }
  mnn="${onnx%.onnx}.mnn"
  if [ -s "$mnn" ] && [ "$mnn" -nt "$onnx" ]; then echo "have: $mnn"; continue; fi
  echo "convert: $(basename "$onnx") -> $(basename "$mnn")"
  mnnconvert -f ONNX --modelFile "$onnx" --MNNModel "$mnn" >/dev/null 2>&1 \
    && echo "  ok" || echo "  FAILED (will fall back to ONNX Runtime)"
done
echo "done."

#!/usr/bin/env bash
# Generate MNN (.mnn) models for the pipeline models MNN runs faster than ONNX Runtime with
# verified parity (all golden tests pass). Each `.mnn` is placed next to its `.onnx`; when
# present (and the binary was built with MNN, ./build.sh --mnn) mlx-mineru uses it, else falls
# back to ONNX Runtime per-model.
#
# Covered (5/8): table-cls (PP-LCNet ~2x), wired-table UNet (~1.15x), OCR det (DBNet ~1.35x),
# OCR rec (SVTR ~1.15x), layout (RT-DETR ~1.6x). NOT covered: slanet (MNN Concat shape bug,
# even in 3.6.0) and mfr (negligible speedup) — they stay on ONNX Runtime.
#
# ocr_det needs graph surgery first: its ONNX head has a NaN/Inf-sanitize subgraph
# (IsInf/IsNaN/Where) MNN can't convert. We cut the model at the head Sigmoid — verified
# identical (y == Sigmoid output) — and the DB post-process consumes that probability map.
#
# Needs pip `mnnconvert` (pip install MNN) + the pipeline models present.
# Usage: ./scripts/convert_pipeline_mnn.sh [<pipeline_dir>]   (default: mumodel/pipeline)
set -euo pipefail
cd "$(dirname "$0")/.."

PIPE="${1:-mumodel/pipeline}"
command -v mnnconvert >/dev/null 2>&1 || { echo "error: mnnconvert not found — pip install MNN" >&2; exit 1; }

# Plain ONNX->MNN conversions.
PLAIN=(
  "$PIPE/TabCls/PP-LCNet_x1_0_table_cls.onnx"
  "$PIPE/TabRec/UnetStructure/unet.onnx"
  "$PIPE/OCR/ocr_rec.onnx"
  "$PIPE/Layout/layout.onnx"
)
for onnx in "${PLAIN[@]}"; do
  [ -f "$onnx" ] || { echo "skip (missing): $onnx"; continue; }
  mnn="${onnx%.onnx}.mnn"
  if [ -s "$mnn" ] && [ "$mnn" -nt "$onnx" ]; then echo "have: $mnn"; continue; fi
  echo "convert: $(basename "$onnx")"
  mnnconvert -f ONNX --modelFile "$onnx" --MNNModel "$mnn" >/dev/null 2>&1 \
    && echo "  ok" || echo "  FAILED (falls back to ONNX Runtime)"
done

# ocr_det: cut the head NaN/Inf-sanitize subgraph, then convert.
det="$PIPE/OCR/ocr_det.onnx"
if [ -f "$det" ]; then
  detmnn="${det%.onnx}.mnn"
  if [ -s "$detmnn" ] && [ "$detmnn" -nt "$det" ]; then echo "have: $detmnn"; else
    echo "convert: ocr_det (cut at head Sigmoid)"
    tmp="$(mktemp -d)/ocr_det_cut.onnx"
    python3 -c "import onnx,sys; onnx.utils.extract_model('$det', '$tmp', ['x'], ['/head/Sigmoid_output_0'])" \
      && mnnconvert -f ONNX --modelFile "$tmp" --MNNModel "$detmnn" >/dev/null 2>&1 \
      && echo "  ok" || echo "  FAILED (falls back to ONNX Runtime)"
  fi
fi
echo "done."

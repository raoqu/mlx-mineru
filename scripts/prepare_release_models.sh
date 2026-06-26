#!/usr/bin/env bash
# Generate ALL Metal/MLX-optimized runtime artifacts into mumodel/pipeline so the PUBLISHED bundle
# carries the accelerated path (getmodel downloads every file in the repo, so once these are
# uploaded, a fresh deploy uses Metal/MLX instead of falling back to the ORT .onnx).
#
# Produces, from the archived source .onnx (orgmodel/) + the base models in mumodel/:
#   OCR/ocr_det.mnn, OCR/ocr_rec.mnn, TabCls/*.mnn, TabRec/UnetStructure/unet.mnn  (pipeline CV)
#   MFR/mfr_encoder.mnn                                  (Swin encoder, Metal ~6.5x)
#   MFR/mfr_decoder.safetensors + mfr_decoder_config.json (mBART decoder, MLX/Metal)
#   Layout/layout_backbone.mnn + Layout/layout_decoder.onnx (RT-DETR backbone split)
#
# Then upload mumodel/ to the HF/ModelScope repo (the same one getmodel pulls from).
# Deps: pip install MNN onnx onnxruntime safetensors numpy.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] pipeline CV models (.mnn) incl. MFR Swin encoder ..."
WITH_MFR=1 bash ./scripts/convert_pipeline_mnn.sh

echo "[2/3] MFR decoder -> MLX weights (.safetensors + _config.json) ..."
python3 ./scripts/extract_mfr_decoder_weights.py >/dev/null && echo "  ok"

echo "[3/3] layout backbone/decoder split (backbone.mnn + decoder.onnx) ..."
python3 ./scripts/split_layout_onnx.py >/dev/null && echo "  ok"

echo
echo "Runtime bundle (mumodel/pipeline):"
find mumodel/pipeline -type f | sort | sed 's/^/  /'
echo
echo "Next: upload mumodel/ to the HF/ModelScope model repo so getmodel ships the accelerated path."
echo "(The .onnx fallbacks are kept for ORT-only builds; drop them from the upload to slim the"
echo " bundle if you only ship Metal/MLX builds — runtime falls back to ORT only when they exist.)"

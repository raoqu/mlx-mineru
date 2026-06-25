#!/usr/bin/env bash
# (Re)generate the pipeline MNN (.mnn) models that mlx-mineru runs instead of ONNX Runtime.
# Reads each source `.onnx` (from orgmodel/, falling back to mumodel/) and writes the `.mnn`
# into the runtime mumodel/. Only needed when updating models — the published mumodel already
# ships these `.mnn`; the matching `.onnx` are archived in orgmodel/ (not loaded at runtime).
#
# 5/8 models run on MNN (golden-verified): table-cls (~2x), layout (~1.6x), ocr-det (~1.35x),
# ocr-rec (~1.15x), wired-table UNet (~1.15x). slanet (MNN Concat bug) and mfr stay on ORT.
# ocr-det needs graph surgery: its head has a NaN/Inf-sanitize subgraph (IsInf/IsNaN/Where)
# MNN can't convert, so cut the model at /head/Sigmoid_output_0 (verified y == Sigmoid).
#
# Needs pip `mnnconvert` (pip install MNN). Usage: ./scripts/convert_pipeline_mnn.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v mnnconvert >/dev/null 2>&1 || { echo "error: mnnconvert not found — pip install MNN" >&2; exit 1; }

SRC="${SRC_DIR:-orgmodel/pipeline}"      # source .onnx (archived originals)
DST="${DST_DIR:-mumodel/pipeline}"        # runtime .mnn destination
FALLBACK="mumodel/pipeline"               # if a source .onnx still lives in mumodel

# relative-path stems (no extension); ocr_det handled separately (needs cutting).
# layout (RT-DETR) is NOT included: MNN's output diverges from ONNX (golden fails) — stays ORT.
PLAIN=( "TabCls/PP-LCNet_x1_0_table_cls" "TabRec/UnetStructure/unet" "OCR/ocr_rec" )

find_onnx() {  # echo the source .onnx for a stem, or ""
  for base in "$SRC" "$FALLBACK"; do [ -f "$base/$1.onnx" ] && { echo "$base/$1.onnx"; return; }; done
}

for stem in "${PLAIN[@]}"; do
  onnx="$(find_onnx "$stem")"; mnn="$DST/$stem.mnn"
  [ -n "$onnx" ] || { echo "skip (no source onnx): $stem"; continue; }
  if [ -s "$mnn" ] && [ "$mnn" -nt "$onnx" ]; then echo "have: $mnn"; continue; fi
  mkdir -p "$(dirname "$mnn")"; echo "convert: $stem"
  mnnconvert -f ONNX --modelFile "$onnx" --MNNModel "$mnn" >/dev/null 2>&1 && echo "  ok" || echo "  FAILED"
done

# ocr_det: cut at the head Sigmoid, then convert.
det="$(find_onnx OCR/ocr_det)"; detmnn="$DST/OCR/ocr_det.mnn"
if [ -n "$det" ] && { [ ! -s "$detmnn" ] || [ "$det" -nt "$detmnn" ]; }; then
  echo "convert: OCR/ocr_det (cut at head Sigmoid)"
  tmp="$(mktemp -d)/cut.onnx"; mkdir -p "$(dirname "$detmnn")"
  python3 -c "import onnx; onnx.utils.extract_model('$det','$tmp',['x'],['/head/Sigmoid_output_0'])" \
    && mnnconvert -f ONNX --modelFile "$tmp" --MNNModel "$detmnn" >/dev/null 2>&1 && echo "  ok" || echo "  FAILED"
fi
echo "done."

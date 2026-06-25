#!/usr/bin/env bash
# Split a downloaded model tree into:
#   mumodel/  — files the runtime actually loads (ONNX + the MLX VLM weights + dicts/tokenizer)
#   orgmodel/ — original pre-conversion files no longer used at runtime (HF/PyTorch safetensors,
#               HF config/tokenizer metadata, duplicate ONNX copies)
#
# The C++ app discovers mumodel/ relative to the working directory or the executable
# (see find_model_root in src/cli/main.cpp), so only mumodel/ needs to ship for inference.
#
# Usage: scripts/split_models.sh [SRC_DIR]   (SRC_DIR defaults to ./models)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-models}"
MU=mumodel
ORG=orgmodel

mv_to() {  # mv_to <dest_root> <relative_path>...
  local dest="$1"; shift
  for rel in "$@"; do
    local from="$SRC/$rel" to="$dest/$rel"
    if [[ -e "$from" ]]; then
      mkdir -p "$(dirname "$to")"
      mv "$from" "$to"
      echo "  $dest/$rel"
    fi
  done
}

echo "[split_models] runtime files -> $MU/"
mv_to "$MU" \
  MinerU2.5-tokenizer/model.safetensors \
  MinerU2.5-tokenizer/vocab.json \
  MinerU2.5-tokenizer/tokenizer.json \
  MinerU2.5-tokenizer/merges.txt \
  pipeline/Layout/layout.onnx \
  pipeline/OCR/ocr_det.onnx \
  pipeline/OCR/ocr_rec.onnx \
  pipeline/OCR/ppocrv6_dict.txt \
  pipeline/MFR/mfr_encoder.onnx \
  pipeline/MFR/mfr_decoder.onnx \
  pipeline/MFR/mfr_vocab.txt \
  pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx \
  pipeline/TabRec/SlanetPlus/slanet-plus.onnx \
  pipeline/TabRec/SlanetPlus/table_structure_dict.txt \
  pipeline/TabRec/UnetStructure/unet.onnx

echo "[split_models] unused pre-conversion originals -> $ORG/"
mv_to "$ORG" \
  MinerU2.5-tokenizer/config.json \
  MinerU2.5-tokenizer/generation_config.json \
  MinerU2.5-tokenizer/preprocessor_config.json \
  MinerU2.5-tokenizer/tokenizer_config.json \
  pipeline/Layout/model.safetensors \
  pipeline/Layout/config.json \
  pipeline/Layout/preprocessor_config.json \
  pipeline/MFR/model.safetensors \
  pipeline/MFR/config.json \
  pipeline/MFR/generation_config.json \
  pipeline/MFR/tokenizer.json \
  pipeline/MFR/tokenizer_config.json \
  pipeline/MFR/special_tokens_map.json \
  pipeline/TabRec/slanet-plus.onnx \
  pipeline/TabRec/unet.onnx

find "$SRC" -type d -empty -delete 2>/dev/null || true
echo "[split_models] done. Runtime: $MU/  Originals (safe to delete): $ORG/"

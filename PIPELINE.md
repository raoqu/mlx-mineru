# Pipeline backend — plan & status

MinerU's classic-CV `pipeline` backend (the non-VLM path): page image → layout
regions → per-region OCR / formula / table recognition → middle_json. Four model
families, ported to a **zero-Python runtime** via **ONNX Runtime (C++)**.

## Strategy (decided)
The custom torch models (layout / OCR / formula) are **exported to ONNX once**
(dev-time, `scripts/export_*_onnx.py`, needs torch + MinerU's modeling code) and
run at runtime via **onnxruntime C++**. The table models are already ONNX.
Pre/post-processing is ported to C++. No Python or torch at runtime.

| Model | Source format | Plan |
|---|---|---|
| Layout — PP-DocLayoutV2 (RT-DETR + reading-order, 53.7M) | safetensors | export → ONNX |
| OCR — DBNet det + CTC rec (PP-OCRv6) | torch/safetensors | export → ONNX (2 models) |
| Formula — UnimerNet (Swin + mBART) | torch `.pth` | export → ONNX (encoder + decoder; autoregressive) |
| Table — SLANet+ / UNet / TabCls | **ONNX** | run as-is |

## Data flow (MinerU `pipeline_analyze` → `model_json_to_middle_json`)
```
page img → [Layout] regions(bbox,type,reading-order)
         → per region: text→[OCR det+rec], formula→[MFR], table→[TabCls→SLANet/UNet]
         → pipeline_magic_model → middle_json → union_make (shared with VLM ✓)
```

## Phases (each: build → verify vs Python golden → commit)
- **P0 — foundation**: vendor onnxruntime C++; `mineru_onnx` wrapper; model-fetch
  scripts. Validate the export path on the hardest model (layout).  ← in progress
- **P1 — Layout**: `layout.onnx` + C++ preprocess (resize 800×800, /255, NCHW) +
  postprocess (per-class thresholds, box scale to page, reading-order decode).
  Golden: Python `PPDocLayoutV2.predict` on demo pages.
- **P2 — Table**: TabCls (wired/simple) → SLANet+ (structure) / UNet; OTSL/HTML
  reuse where possible. Golden: Python table rec on cropped tables.
- **P3 — OCR**: DBNet detection (→ text-line boxes, DB postprocess) + CTC rec
  (→ text, CTC greedy decode + dict). Golden: Python OCR on text crops.
- **P4 — Formula (MFR)**: Swin encoder + mBART decoder (autoregressive ONNX
  generation in C++). Golden: Python UnimerNet on formula crops.
- **P5 — Assembly**: port `pipeline_magic_model` + `model_json_to_middle_json`
  (para split, spans, caption assoc) → middle_json → existing `union_make`.

## Scope note
Each model family is comparable in size to the Qwen2-VL port. This is the single
largest remaining task; it advances phase by phase with golden verification.

## Status
- P0: onnxruntime selected; layout model downloaded; export script written
  (`scripts/export_layout_onnx.py`) — validating clean ONNX export (the gate for
  the whole strategy).

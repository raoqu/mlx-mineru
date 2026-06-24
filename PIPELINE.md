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
- **P2 — Table** (TabCls ✅): TabCls (wired/simple) → SLANet+ (structure) / UNet; OTSL/HTML
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
- **P0 gate PASSED ✅**: PP-DocLayoutV2 (RT-DETR + reading-order) exports to
  `layout.onnx` (215MB) via `scripts/export_layout_onnx.py`. Validation on a real
  page: onnxruntime == torch — order_logits max|diff| 2.2e-4, and the
  **thresholded detections match exactly** (the raw 300-query tensor diff is only
  tie-broken low-confidence padding rows, which get thresholded out). The
  `take_along_dim -> gather` monkey-patch was needed for the export (opset 17).
- **P1 foundation ✅**: ONNX Runtime C++ vendored (dylib from pip package,
  headers from v1.26.0 tag; `scripts/fetch_onnxruntime.sh`). CMake imported
  `onnxruntime` target. `ctest onnx_smoke` loads `layout.onnx` in C++ and runs it
  — output shapes correct (logits 1x300x25, pred_boxes 1x300x4, order_logits
  1x300x300). The pipeline runtime path is proven end-to-end in C++.
- **P1 layout detector ✅**: `LayoutDetector` (`src/pipeline/layout_det.cpp`) —
  onnxruntime inference + core post-process (box decode -> scale -> sigmoid ->
  topk -> conf>=0.45 -> clip). `ctest layout_det` matches the Python core golden
  EXACTLY (same labels/boxes/scores on a real page). New `mineru_pipeline` lib.
- **P2 TabCls ✅**: `TableClassifier` (`src/pipeline/table_cls.cpp`) — bilinear
  resize-256 + center-crop-224 + ImageNet-normalize + ONNX; `ctest tab_cls`
  matches a Python golden (same algorithm) exactly. Added reusable
  `resize_bilinear_rgb8`. (cv2 fixed-point bilinear approximated by float bilinear;
  doesn't flip the class.)
- **P3 OCR export gate ✅**: PP-OCRv6 det (DBNet) + rec (CTC) -> ocr_det.onnx /
  ocr_rec.onnx via `scripts/export_ocr_onnx.py`. Built directly through
  pytorchocr BaseOCRV20 + arch_config.yaml + RepVGG reparameterization (avoids the
  cv2/shapely/pyclipper deps the full import needs). torch-vs-onnxruntime:
  det 3.9e-5, rec 7.9e-5. Dict ppocrv6_dict.txt (18709 chars) copied.
- **P3 OCR rec ✅**: `TextRecognizer` (`src/pipeline/ocr_rec.cpp`) — bilinear
  resize-to-H48 + /127.5-1 pad + ocr_rec.onnx + CTC greedy decode (drop blank,
  collapse repeats) + 18710-char dict. `ctest ocr_rec` reads real Chinese from an
  a.pdf line ("3. 了解组织行为学学科体系...") matching the Python golden exactly.
- **P3 OCR det post-process ✅**: `db_postprocess` (`src/pipeline/ocr_det.cpp`) —
  faithful port of MinerU `DBPostProcess` (box_type="quad"): binarize>0.3 -> 8-conn
  components -> convex hull (Andrew) -> minAreaRect (rotating calipers) ->
  get_mini_boxes ordering -> box_score_fast (int-vertex scanline fill, skip<0.6) ->
  unclip (rectangle expand by area*1.5/perimeter, no Clipper needed) -> scale with
  np.round (half-to-even). `ctest ocr_det` matches MinerU's real `DBPostProcess` on
  the same prob-map: **19/19 boxes, ≤1px per coordinate** (residual is float32
  minAreaRect vs OpenCV, sub-pixel). Golden: `scripts/gen_ocr_det_golden.py` saves
  the prob-map + real cv2/pyclipper boxes, isolating geometry from the resize.
- **P3 OCR det end-to-end ✅**: `TextDetector` (`src/pipeline/ocr_det.cpp`) —
  DetResizeForTest (limit_type="max" 960, round/32, half-to-even) + NormalizeImage
  (mean/std, RGB, CHW) + ocr_det.onnx + db_postprocess. `ctest ocr_det` runs the full
  C++ path on the saved a.pdf render: **19/19 boxes match MinerU within ≤1px/coord**
  (cv2.resize vs our half-pixel bilinear is the only residual). Detector takes RGB +
  (w,h), returns text-line quads in source pixels — ready to crop and feed rec.
- **Next OCR (C++)**: chain det->crop(perspective/rotate)->rec for full page text
  (MinerU `get_rotate_crop_image`), then per-region OCR in the pipeline assembly.
- **Also queued**: P2 SLANet+/UNet table *structure* recognition (the table HTML); the
  layout heuristic-filter layer + reading order; then OCR (P3), formula (P4),
  assembly (P5).

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
- **P3 OCR full-page chain ✅**: `OcrPipeline` (`src/pipeline/ocr.cpp`) — faithful port
  of MinerU `PytorchPaddleOCR.__call__`: det -> `sorted_boxes` -> `merge_det_boxes`
  (calculate_is_angle / merge_spans_to_line / merge_overlapping_spans) -> rotate-crop
  (`get_rotate_crop_image`, aligned fast-path + perspective + np.rot90 for tall crops) ->
  batched CTC rec (sort by aspect, batch of 6 sharing the widest crop's `max_wh_ratio`)
  -> drop_score 0.5. `ctest ocr_page` runs the whole C++ chain on a.pdf p0 vs MinerU's
  real chain (its actual `sorted_boxes`/`merge_det_boxes`/`get_rotate_crop_image_for_text_rec`
  + predict_rec batching): **19/19 lines, boxes within ~1px, 18/19 texts exact**. The one
  text delta is a doubled em-dash: we don't link OpenCV, so our bilinear differs from
  cv2.resize by ≤1 LSB (cv2 isn't bit-exact across its own scalar/SIMD builds either), and
  that sub-pixel delta shifts a det box ~1px which re-buckets one crop's rec batch.
- **Next OCR (C++)**: integrate per-region OCR into the pipeline assembly (P5); the chain
  is ready to run on layout text/title regions.
- **P4 formula (MFR) ✅**: UniMERNet (UnimerSwin encoder + UnimerMBart decoder) exported
  to two ONNX graphs (`scripts/export_mfr_onnx.py`): `mfr_encoder.onnx`
  (pixel[1,3,192,672]→hidden[1,126,768]) + `mfr_decoder.onnx` (no KV cache: input_ids +
  hidden→logits, recompute each step — formulas are short). `take_along_dim→gather`
  patch for opset-17. Greedy decode bit-exact vs torch (86/86 ids incl EOS). `FormulaRecognizer`
  (`src/pipeline/formula_rec.cpp`): UniMERNet preprocess (crop-margin via min-max+<200
  threshold bbox, aspect resize, center pad, cv2 RGB2GRAY + `(g-0.7931·255)/(0.1738·255)`)
  + encoder + greedy loop + byte-level BPE decode (GPT-2 byte map + vocab, no merges needed
  for decode — matches HF `decode(skip_special_tokens)` exactly). `ctest mfr`: model path
  (pixel→ids/latex) **exact**, end-to-end (raw crop→latex) **also exact** on a rendered
  formula. Models gitignored; goldens (pixel, ids, latex, input) committed.
- **P2 table structure (SLANet+) ✅**: `TableRecognizer` (`src/pipeline/table_rec.cpp`) —
  faithful port of MinerU slanet_plus: TablePreprocess (resize-488 + BGR normalize + pad)
  + slanet-plus.onnx (loc_preds[L,8] + structure_probs[L,50]) + TableLabelDecode (per-step
  argmax, `<td>` bbox decode) + `adapt_slanet_plus` + `TableMatch` (get_boxes_recs,
  filter-by-min-y, per-OCR-box best cell by (1-IoU, distance), HTML assembly with
  colspan/rowspan + `<b>` handling). `ctest table`: structure tokens **exact**, cells
  within **0.03px**, and full **HTML matches MinerU exactly** (OCR'd the table with the
  onnx det+rec chain, fed the same ocr_result to MinerU's TableMatch for the golden).
  Wired-table UNet path is the remaining table variant.
- **P5 assembly — golden captured, port pending**: MinerU's real pipeline backend now
  runs locally and `scripts/gen_pipeline_golden.py` dumps the per-page **model_list**
  (layout_dets: region boxes {text,paragraph_title,...} + `ocr_text` line boxes carrying
  OCR text) and the assembled **middle_json** (`tests/golden/pipeline_*.json`, a.pdf 3
  pages). Setup notes (config, deps, the Layout config.json that the raw HF download
  omits) are in the script header. The transform to port: `MagicModel`
  (`__fix_axis` scale-by `model_w/page_pt_w` ≈ 2.78 @200dpi → page-point coords; group
  `ocr_text` spans into region blocks via SpanBlockMatcher; `merge_spans_to_line` +
  sort; label→BlockType map) + `model_json_to_middle_json` + `para_split` + title
  leveling → `para_blocks`; then the existing byte-exact `union_make` renders Markdown.
  a.pdf p0 is text/title-only (29 dets → 10 para_blocks, 1:1) — the natural first slice.
- **P5 assembly — text/title slice ✅**: `assemble_page_info` (`src/core/pipeline_assemble.cpp`)
  ports the MagicModel text path: `__fix_axis` (scale by model_w/page_w, int-truncate, drop
  ≤2px), `__post_process` (ocr_text → spans, region dets reindexed 1..N), label→BlockType
  map, greedy span→block match (overlap-in-span-area > 0.5), `merge_spans_to_line` +
  left-to-right sort, `_post_block_process` (doc_title→title/level1, paragraph_title→
  title/level2, vertical_text→text), discarded split, and `para_split`'s `bbox_fs`. `ctest
  pipeline_assemble`: **all 10 blocks structurally exact vs MinerU** (bbox, index, type,
  level, lines, bbox_fs) — only span text is left blank, since MinerU fills it in a separate
  post-OCR step (re-running OCR rec on each span crop), which ties to the existing C++ OCR
  recognizer. Next: wire post-OCR text-fill + image/table/formula spans → first full
  pipeline-backend page → Markdown via the byte-exact union_make.
- **P5 post-OCR text-fill ✅**: `fill_span_text` (`src/pipeline/post_ocr.cpp`) — faithful
  port of `_apply_post_ocr`: crop each span from the page image (bbox×scale, rotate-if-tall
  via `get_rotate_crop_image`), batched rec (sort by aspect, batch-of-6 shared max_wh),
  drop below `min_confidence` 0.5. `ctest pipeline_textfill` renders a.pdf p0 at 200 DPI via
  the byte-exact `PdfDocument`, fills all 19 spans, **ASCII (digits/punct) exact vs MinerU**.
  Finding: **a.pdf is a digital PDF** — MinerU reads its embedded text layer via `pdftext`
  (Kangxi-radical codepoints + NBSP), so the CJK is NFKC-equivalent to our OCR-read glyphs
  (CJK-unified) but not byte-equal. The OCR text-fill is the scanned-doc path and is correct;
  faithful digital-PDF char extraction (pdftext + fill_char_in_spans) is a separate follow-up.
  With text filled, the existing byte-exact union_make renders the page → Markdown.
- **P5 full vertical ✅ (first pipeline-backend page → Markdown)**: `ctest pipeline_md` runs
  the whole chain for a.pdf p0 — render (PdfDocument 200dpi) → `assemble_page_info`
  (model_list → blocks) → `fill_span_text` (post-OCR) → `union_make` → Markdown — and the
  output (## headings + paragraphs, correct reading order + readable text) is **ASCII-exact
  vs `union_make` on MinerU's golden middle_json**. The remaining work to a faithful CLI is
  the per-page driver loop + the non-text paths (digital pdftext extraction, visual spans,
  UNet tables).
- **Also queued**: UNet wired-table structure; digital-PDF text extraction (pdftext) for
  text-layer PDFs; visual-span path (image/table/formula) in the assembly; para_split
  cross-block merging for multi-block paragraphs.

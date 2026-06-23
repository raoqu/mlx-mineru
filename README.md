# mlx-mineru

C++ native reimplementation of [MinerU](https://github.com/opendatalab/MinerU) `3.4.0`
document parsing (PDF / image / Office → Markdown / JSON), **with no Python runtime**,
MLX/Metal-accelerated on Apple Silicon.

See [GOAL.md](GOAL.md), [PLAN.md](PLAN.md), and [AGENT.md](AGENT.md) for scope, plan,
and working principles.

## Build & test

```bash
./scripts/build_and_test.sh
```

Requires CMake ≥ 3.20 and a C++20 compiler (Apple clang). Header-only dependencies
(`nlohmann/json`, `CLI11`) are vendored under `third_party/`.

## CLI (early)

```bash
./build/mlx-mineru -p demo.pdf --page 0 -o layout.json
```

Renders a PDF page and runs the Qwen2-VL **layout-detection** step natively on
Apple Silicon (MLX/Metal) — no Python at runtime — emitting the detected document
structure (block types + bboxes). On a sample page it detects headers/title/body/
footnotes in reading order. Per-block content extraction → `middle_json` → Markdown
are the in-progress next steps (see AGENT.md). Note: generation currently uses a
KV-cache-less path (~38s/page); a KV cache is the next optimization.

## Status

Under construction, phase by phase (see PLAN.md). Every phase is built, tested, and
committed before the next begins.

- **Phase 0 — core data model** ✅ `middle_json` typed model + enums + lossless JSON
  round-trip (`ctest`: `middle_json_roundtrip`).
- **Phase 1 — output contract** ✅ `union_make` (mm/nlp markdown + content_list v1/v2),
  faithful port of MinerU's renderer. Verified against golden output produced by the
  real Python `union_make` (`scripts/gen_golden.py`; `ctest`: `mkcontent_golden`).
- **Phase 3 — PDF rasterization** ✅ pdfium-backed `PdfDocument::render_page`, faithful
  port of MinerU's `page_to_image` (scale=dpi/72 capped to 3500px long side, ceil sizing).
  Verified against `pypdfium2` golden — exact dimensions/scale per page (`scripts/gen_pdf_golden.py`;
  `ctest`: `pdf_raster`). pdfium binary fetched via `scripts/fetch_pdfium.sh`.
- **Phase 4a — MLX/Metal toolchain** ✅ MLX C++ linked from the pip package; GPU
  compute smoke test (`ctest`: `mlx_smoke`).
- **Phase 4b — Qwen2 tokenizer** ✅ byte-level BPE with the exact Qwen2 pre-tokenizer
  (Unicode tables generated from `unicodedata`). Exact encode/decode parity with HF
  `transformers` (`scripts/gen_tokenizer_golden.py`; `ctest`: `tokenizer`).
- **Phase 4c — image preprocessing** ✅ Qwen2VLImageProcessor port (smart_resize +
  PIL bicubic + normalize + patchify). Bit-exact vs transformers (`ctest`: `preprocess`).
- **Phase 4d (LLM) — Qwen2-VL language model in MLX C++** ✅ full decoder (GQA, MRoPE,
  RMSNorm, SwiGLU, tied embeddings) loading the real safetensors. Verified vs the
  transformers model: exact next-token argmax, top-10 logits within tolerance, greedy
  continuation matching (near-tie aware) — `scripts/gen_llm_golden.py`; `ctest`:
  `llm_forward`. Weights via `scripts/fetch_weights.sh` (~2.2GB, gitignored).
- **Phase 4d (vision + multimodal)** ✅ Qwen2-VL vision tower + 3D-MRoPE multimodal
  merge + greedy generation in MLX C++. Verified vs transformers: vision embeds
  (`ctest`: `vision`) and full end-to-end image→text generation (`ctest`: `vlm`).
- **Phase 4e (layout parser)** ✅ parser for the model's layout-detection output
  (`<|box_start|>…<|ref_*|>…` grammar, bbox conversion, table-internal filter),
  matching mineru-vl-utils (`ctest`: `layout`).

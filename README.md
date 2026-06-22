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

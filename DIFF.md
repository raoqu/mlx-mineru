# DIFF — mlx-mineru vs. MinerU (Python, `~/research/MinerU`, 3.4.0)

> ⚠️ **SUPERSEDED for the pipeline backend** — this doc predates the native
> pipeline-backend work (layout/OCR/formula/table → middle_json → Markdown is now
> implemented). See [GAP.md](GAP.md) for the current, accurate gap analysis. The VLM
> sections below remain valid.

Functional differences between this C++/MLX reimplementation and upstream MinerU.
Legend: ✅ parity (verified against the source/reference) · 🟡 simplified/partial ·
❌ not implemented.

> One-line summary: mlx-mineru is a **native, zero-Python, Apple-Silicon
> reimplementation of MinerU's *VLM* pipeline for PDFs**. The output contract, PDF
> rasterization, image preprocessing, tokenizer, OTSL→HTML, and a subset of
> post-processing are faithful; the classic CV/Office backends and several
> document-assembly refinements are not (yet) ported.

---

## 1. Backends

| Backend | MinerU | mlx-mineru |
|---|---|---|
| `vlm` (Qwen2-VL) | ✅ engines: `transformers` / `vllm` / `mlx-engine` / `http-client` | ✅ **native MLX C++** (own engine) |
| `pipeline` (7 classic CV models: layout/MFR/OCR/table) | ✅ | ❌ |
| `hybrid` (pipeline + vlm) | ✅ (CLI default) | ❌ |
| `office` (docx/pptx/xlsx, zip+XML, no ML) | ✅ | ❌ |

mlx-mineru implements **only the VLM backend**. There is no pipeline (DBNet/CTC OCR,
UnimerNet formula, SLANet table, RT-DETR layout), no hybrid, no Office path.

## 2. Input formats

| Input | MinerU | mlx-mineru |
|---|---|---|
| PDF | ✅ | ✅ (pdfium) |
| Images (png/jpg/jp2/webp/gif/bmp/tiff) | ✅ | ❌ |
| Office (docx/pptx/xlsx) | ✅ | ❌ |

## 3. VLM pipeline — what IS faithful (✅, verified)

| Stage | Notes / verification |
|---|---|
| PDF rasterization | `scale=dpi/72`, long side capped 3500, `ceil` sizing — exact dims vs `pypdfium2` (`pdf_raster` test) |
| Qwen2-VL two-step extract | layout-detect prompt → per-block `Table/Formula/Image/Text Recognition` prompts (`mineru-vl-utils` contract) |
| Image preprocessing | smart_resize + PIL bicubic + CLIP normalize + patchify — **bit-exact** vs transformers `Qwen2VLImageProcessor` (`preprocess` test) |
| Qwen2 tokenizer | byte-level BPE + Qwen2 pre-tokenizer — exact ids vs HF `transformers` (`tokenizer` test) |
| Qwen2-VL model | vision ViT + LLM (GQA, mRoPE, RMSNorm, tied embeds) in MLX — logit/feature parity (`llm_forward`, `vision`, `vlm` tests) |
| Output renderer `union_make` | `mm_markdown` / `nlp_markdown` / `content_list` / `content_list_v2` — **byte/semantic-exact** vs Python `union_make` (`mkcontent_golden`) |
| Tables OTSL → HTML | `convert_otsl_to_html` (rowspan/colspan/escape) — exact (`otsl` test) |
| Text/equation post_process | `display→inline`, macro spacing, underscores, delimiter/brace fixes — exact (`post_process` test) |
| `middle_json` contract | field-compatible; unknown keys preserved (`middle_json_roundtrip`) |

## 4. VLM pipeline — simplifications & gaps (🟡 / ❌)

| Area | MinerU | mlx-mineru |
|---|---|---|
| **Sampling** | per-type `SamplingParams` (presence/frequency penalty, `no_repeat_ngram`) | 🟡 **greedy only** (model's `top_k=1`); repetition-penalty / no-repeat-ngram NOT applied — may loop/diverge on repetitive tables vs MinerU |
| **MagicModel** (`vlm_magic_model.py`, ~850 lines): block association, reading order, caption/footnote ↔ image/table/code grouping, title levels | full | 🟡 basic type mapping only; **title heading level fixed at `1`**; captions/footnotes not associated into composite blocks |
| **Rotation tokens** `<|rotate_*|>` | applied (deskew) | ❌ ignored |
| **post_process equation fixers** | 7 (`left_right`, `big`, `leq`, `eqqcolon`, `double_subscript`, delimiters, braces) | 🟡 2 ported (delimiters, unbalanced braces) |
| **Image reclassification** (`process_image_or_chart`: pure_table / pure_formula / chart sub-class / natural_image) | ✅ | ❌ image/chart kept as-is |
| **Cross-page table merge** | ✅ | ❌ |
| **Mermaid / chart repetition fixes**, `[Non-Text]` placeholder handling | ✅ | ❌ |
| **`detect_lang`** (CJK vs Western spacing) | `fast-langdetect` (fasttext) | 🟡 CJK-script heuristic (agrees on unambiguous text) |
| **Batched two-step extract** | batched | 🟡 per-block vision; length-bucketed batched generation (`--batch`) |
| **Layout image resolution** | MinerU's own resize | 🟡 fixed 1036×1036 for layout pass |

## 5. Output

| | MinerU | mlx-mineru |
|---|---|---|
| Dir layout | `<out>/<name>/<method>/` (`auto`/`txt`/`ocr`/`vlm`) | `<out>/<name>/vlm/` |
| `<name>.md` | ✅ | ✅ |
| `<name>_content_list.json` | ✅ | ✅ |
| `<name>_middle.json` | ✅ | ✅ |
| `<name>_model.json` (raw model output) | ✅ | ❌ |
| `content_list_v2` | optional | 🟡 implemented in `union_make`, not written by the CLI |
| `images/` crops | ✅ | ✅ (JPEG via stb; hash filenames differ from MinerU's) |
| Visualization PDFs (span/layout draw) | ✅ | ❌ |

Content types in the Markdown: text, titles, tables (OTSL→HTML), equations (`$$`),
images/charts (`![]()` + `<details>`); headers/footers/page-numbers → `discarded_blocks`.

## 6. CLI flags

| MinerU (`mineru`) | mlx-mineru (`mlx-mineru`) |
|---|---|
| `-p/--path`, `-o/--output` | `-p/--path`, `-o/--output` |
| `-b/--backend` (pipeline/vlm/hybrid…) | — (vlm only) |
| `-m/--method` (`auto`/`txt`/`ocr`) | `-m/--model` (**model directory**, different meaning) |
| `-l/--lang`, `-f/--formula`, `-t/--table`, `--effort`, `--image-analysis` | ❌ |
| `-s/--start`, `-e/--end` | ✅ `-s`, `-e` |
| `-u/--url`, `--api-url`, `--client-side-output-generation`, `-d` device, `-v` version | ❌ |
| — | ✅ `--bits` (4-bit quant), `--batch`, `--layout-only`, `--server`, `--host`, `--port` |

## 7. HTTP API (`--server` / `fast_api.py`)

| | MinerU (protocol v2) | mlx-mineru |
|---|---|---|
| `GET /health` | ✅ | ✅ |
| `POST /file_parse` | rich multipart: `files`, `lang_list`, `backend`, `parse_method`, `formula/table_enable`, `return_md/middle_json/content_list/images`, `response_format_zip` | 🟡 PDF body **or** multipart `files` → `{md_content, content_list}` only |
| Async `POST /tasks`, `GET /tasks/{id}`, `GET /tasks/{id}/result` | ✅ | ❌ (synchronous only) |
| `X-MinerU-Task-*` headers, ZIP responses, image return | ✅ | ❌ |

## 8. Runtime & dependencies

| | MinerU | mlx-mineru |
|---|---|---|
| Language / runtime | Python 3 + PyTorch | **native C++; no Python at runtime** |
| Accelerator | CUDA / MPS / CPU (torch); vllm optional | **MLX/Metal** (Apple Silicon) |
| Table models | ONNX Runtime (pipeline) | n/a (VLM emits OTSL) |
| Model weights | per-backend (VLM + 7 pipeline models) | single VLM `model.safetensors` (4-bit quantized at load by default) |
| Config | `~/mineru.json`, many env vars | compiled-in `Qwen2VLConfig` + CLI flags |

## 9. Fidelity caveats (not byte-for-byte vs MinerU's *running* VLM)

Each component is verified against its **upstream reference** (transformers preproc/
tokenizer, pypdfium2, `union_make`, `otsl2html`, `post_process`) — but the **end-to-end
VLM output is not byte-aligned to MinerU's actual runs**, because:

- **greedy vs sampled** decoding (no penalties / no-repeat-ngram);
- **4-bit quantization** by default (`--bits 0` for full bf16) and **batched** decoding
  (`--batch 1` for deterministic) both perturb rare near-tie tokens;
- PDF pixels differ slightly (pdfium build 7906 vs pypdfium2's 7891 — anti-aliasing);
- `detect_lang` heuristic vs fast-langdetect.

Reproducing MinerU's exact VLM tokens would require running their Python VLM to mint
golden output, which isn't available in this environment.

## 10. Summary table

| Capability | MinerU | mlx-mineru |
|---|---|---|
| PDF → Markdown (VLM) | ✅ | ✅ |
| Tables / equations / images in MD | ✅ | ✅ |
| Office / image inputs | ✅ | ❌ |
| Pipeline / hybrid backends | ✅ | ❌ |
| Full doc assembly (captions, levels, reading order) | ✅ | 🟡 |
| Sampling / repetition control | ✅ | ❌ (greedy) |
| Async task API, rich `/file_parse` | ✅ | 🟡 |
| Zero-Python native runtime | ❌ | ✅ |
| MLX/Metal optimization (quant + batching) | n/a | ✅ |

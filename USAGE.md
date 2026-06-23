# mlx-mineru — Usage

Native C++/MLX reimplementation of MinerU's VLM pipeline: **PDF → Markdown** on
Apple Silicon (MLX/Metal), with **no Python at runtime**. This document covers
how to build, run, the HTTP API, and the model / native-library files it needs.

---

## 1. Requirements

| | |
|---|---|
| Platform | macOS on Apple Silicon (arm64) |
| Build toolchain | CMake ≥ 3.20, a C++20 compiler (Apple clang) |
| Build/setup only | Python 3 with the `mlx` pip package (used to *vendor* the MLX C++ library and to generate test goldens). **Not needed to run the binary.** |
| Runtime | The vendored native libs in `third_party/` + Apple system frameworks. No Python interpreter, no `.py`. |

> The pip `mlx` package is only a convenient *source* of the prebuilt MLX C++
> runtime. After `build.sh`, the binary loads `libmlx.dylib`, `libjaccl.dylib`,
> `mlx.metallib`, and `libpdfium.dylib` from `third_party/` — deleting the
> Python install does not affect it.

---

## 2. Build

```bash
./build.sh             # fetch deps + configure + build (Release)
./build.sh --test      # build, then run the test suite (12 tests)
./build.sh --weights   # also download the ~2.2 GB model weights
./build.sh --debug     # Debug build
./build.sh --clean     # remove build/ first (fresh configure)
./clean.sh             # remove build/ and output/
./clean.sh --deps      # also remove fetched native deps (third_party/{mlx,pdfium})
./clean.sh --all       # also remove the model dir (models/…, ~2.2 GB)
```

`build.sh` runs the dependency fetchers (below) on demand, then builds the
libraries, the `mlx-mineru` CLI, and the tests. The binary is `build/mlx-mineru`.

---

## 3. Dependency files

All of these are **gitignored** and fetched by scripts (called automatically by
`build.sh`). They split into *native libraries* (needed to run) and *model files*
(needed to run the VLM).

### 3a. Native libraries → `third_party/`

| File | Size | Fetched by | Notes |
|---|---|---|---|
| `third_party/pdfium/lib/libpdfium.dylib` | 7.4 MB | `scripts/fetch_pdfium.sh` | PDF rasterization (bblanchon prebuilt) |
| `third_party/mlx/lib/libmlx.dylib` | 21 MB | `scripts/fetch_mlx.sh` | MLX C++ tensor runtime |
| `third_party/mlx/lib/libjaccl.dylib` | 0.9 MB | `scripts/fetch_mlx.sh` | MLX private dependency |
| `third_party/mlx/lib/mlx.metallib` | 150 MB | `scripts/fetch_mlx.sh` | Metal GPU shaders (loaded next to libmlx) |
| `third_party/mlx/include/` | — | `scripts/fetch_mlx.sh` | MLX headers (build only) |

Header-only deps (`nlohmann/json`, `CLI11`, `cpp-httplib`, `stb`) are committed
under `third_party/` and need no fetch.

### 3b. Model files → `models/MinerU2.5-tokenizer/` (default `--model` dir)

Model: **`opendatalab/MinerU2.5-Pro-2605-1.2B`** (Qwen2-VL, ~1.2 B params).

```bash
./scripts/fetch_weights.sh      # weights + tokenizer + configs (~2.2 GB, resumable)
./scripts/fetch_tokenizer.sh    # tokenizer + configs only (~16 MB; no weights)
```

| File | Required at runtime | Used for |
|---|---|---|
| `model.safetensors` (2.2 GB) | ✅ | model weights (`mx::load_safetensors`) |
| `vocab.json` | ✅ | tokenizer vocab |
| `merges.txt` | ✅ | tokenizer BPE merges |
| `tokenizer.json` | ✅ | added/special tokens |
| `tokenizer_config.json` | optional | — |
| `config.json`, `generation_config.json`, `preprocessor_config.json` | optional | model/preproc params are compiled-in defaults; these are kept for reference |

> The C++ does **not** read `config.json` at runtime — the Qwen2-VL config
> (hidden 896, 24 layers, GQA 14:2, vision depth 32, etc.) is baked into
> `Qwen2VLConfig`. Point `--model` at any dir containing the four required files.

---

## 4. CLI usage

```
mlx-mineru -p <file.pdf> [options]
```

| Flag | Default | Description |
|---|---|---|
| `-p, --path <pdf>` | — | Input PDF (required unless `--server`) |
| `-m, --model <dir>` | `models/MinerU2.5-tokenizer` | Model dir (weights + tokenizer) |
| `-s, --start <n>` | `0` | First page (0-based) |
| `-e, --end <n>` | last | Last page (0-based, inclusive) |
| `-o, --output <dir>` | `output` | Output directory |
| `--layout-only` | off | Only run layout detection, emit JSON |
| `--server` | off | Run the HTTP API server instead of converting |
| `--host <h>` | `127.0.0.1` | Server bind host |
| `--port <p>` | `8000` | Server port |

### Examples

```bash
# Whole PDF -> Markdown (+ content_list + middle json)
./build/mlx-mineru -p paper.pdf -o output

# Just pages 0..4
./build/mlx-mineru -p paper.pdf -s 0 -e 4 -o output

# Layout detection only (block types + bboxes as JSON)
./build/mlx-mineru -p paper.pdf --layout-only -o output

# Custom model location
./build/mlx-mineru -p paper.pdf -m /path/to/MinerU2.5-tokenizer -o output
```

### Output layout (`-o <out>`)

```
<out>/<pdf-stem>/vlm/
├── <stem>.md                    # Markdown (union_make, MM_MD)
├── <stem>_content_list.json     # content_list (union_make)
├── <stem>_middle.json           # full middle_json
├── images/                      # cropped image/chart regions (JPEG)
│   └── <hash>.jpg
└── <stem>_layout.json           # ONLY with --layout-only
```

**Content handled:** text (+post_process), titles (`#`), tables (OTSL→HTML with
rowspan/colspan), equations (`$$`, with delimiter/brace fixes), images & charts
(saved as JPEG, referenced `![](images/…)` with a `<details>` analysis block).
Headers/footers/page-numbers go to `discarded_blocks`.

Progress and timing are printed to **stderr**; on a typical page expect a few
seconds (KV-cached generation).

---

## 5. HTTP API (`--server`)

```bash
./build/mlx-mineru --server --host 127.0.0.1 --port 8000
# loads the model once, then serves requests (serialized through the model).
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### `POST /file_parse`

Accepts a PDF as the **raw request body** (`Content-Type: application/pdf`) or as
**multipart form-data** under the field name `files`. Converts the whole document.

```bash
# raw body
curl -X POST --data-binary @paper.pdf \
     -H "Content-Type: application/pdf" \
     http://127.0.0.1:8000/file_parse

# multipart
curl -X POST -F "files=@paper.pdf" http://127.0.0.1:8000/file_parse
```

**Response** `200 application/json`:

```json
{
  "md_content": "# Title\n\n…markdown…",
  "content_list": [ { "type": "text", "text": "…", "page_idx": 0 }, … ]
}
```

**Errors:** `400` if the body is not a PDF (must start with `%PDF-`); `500` with
`{"error": "..."}` on a processing failure.

> Server mode does not write image files to disk, so images are omitted from the
> returned Markdown (text/tables/equations are included). Use the CLI for full
> output including saved image crops.

---

## 6. Notes & limitations

- **Greedy decoding** (`top_k=1`), matching the model's generation config.
- **Simplified vs MinerU Python** (non-blocking): title heading levels are all
  `#`; MagicModel caption/footnote association, the full equation-fix set, image
  reclassification, and cross-page table merge are not yet ported. Core output is
  faithful (`union_make`, OTSL→HTML, text/equation post_process are verified
  byte/parity-exact against the source).
- **Verification:** `./build.sh --test` runs 12 tests checking parity against the
  real transformers / pypdfium2 / mineru-vl-utils references.

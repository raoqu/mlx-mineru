# Performance & Optimization Plan (MLX/Metal)

Native C++/MLX, Apple Silicon. Every one-shot CLI run prints a per-phase profile
(model load, pdf rasterize, layout vision/generate, content vision/generate,
assembly, overhead) — use it to target work.

## Where the time goes (measured)

`a.pdf` (3 pages, 26 blocks, default `--bits 4 --batch 6`): **11.7s total**

| Phase | Share |
|---|---:|
| layout vision | 24.6% |
| content vision | 30.8% |
| layout generate | 20.4% |
| content generate | 19.1% |
| load / raster / assembly / overhead | ~5% |

- **~95% is the Qwen2-VL forward passes.** All the ported C++ glue (rasterize,
  `union_make`, tokenizer, JSON, image save) is <5% combined — not worth optimizing.
- **Vision (ViT) = 55%**, generation = 40%. The balance is **document-dependent**:
  text-heavy docs (demo1) are generation-bound (~64%); figure/short-text docs
  (a.pdf) are vision-bound (~55%).
- **Content vision = 26 *sequential* crop encodes** — the largest single tractable
  slice and the top target.

## Already done

- **4-bit LLM quantization** (`--bits`) — decoder is memory-bandwidth bound; ~4× less
  weight traffic per token. Vision left bf16 (compute-bound; quantizing it measured
  *slower*).
- **Length-bucketed batched generation** (`--batch`) — content gen 259→~413 tok/s.
- **Single-sync content vision** (`forward_vision_batch`) — all crops built lazily and
  eval'd in one batch (one GPU sync vs one per crop). Output verified identical.
- **Cross-page batched layout** (`convert_batched`) — layout images are uniform
  1036×1036, so all pages' layout vision + generation run in one batched pass. Layout
  *generation* (batch-1, memory-bandwidth bound page-by-page) sped up **2.38s→1.45s
  (239→394 tok/s)** on a.pdf (3 pages); total **11.7s→10.75s**, output byte-identical.
  Scales with page count. (Layout vision stayed flat — compute-bound, as predicted.)

## Findings — vision batching (Metal/MLX)

Several vision-batching strategies were implemented and measured on `a.pdf`:

| Approach | content vision | verdict |
|---|---:|---|
| per-crop (baseline) | 3.60s | — |
| lazy + single eval | 3.57s | neutral (MLX runs forwards serially; compute-bound) |
| packed: concat patches + **dense** block-diagonal mask | 3.96s | **slower** — O(T²) attention waste > GEMM-batching gain; reverted |
| **batch dim, grouped by identical grid** (`forward_vision_uniform`) | **3.40s** | **~5% (kept)** — same-grid crops share one big Metal GEMM, attention stays per-image (no mask/waste) |

For the uniform **layout** images (all same grid) the batch-dim path is exact-batched
but **flat** (2.88s) — a single 5476-patch forward already saturates the Metal GEMM, so
B=3 adds no occupancy.

**Conclusion:** vision is compute-bound and MLX's Metal GEMM is already near-optimal.
Batching helps only marginally (content crops sharing a grid); large images are already
saturated. The remaining lever — a custom **varlen/`cu_seqlens` flash-attention Metal
kernel** — would not help here either, because for these crop sizes the *linear layers*
(GEMM, already optimal), not attention, dominate. The kept `forward_vision_uniform`
(batch-dim, grid-grouped) is correct (output byte-identical) and is the right structure;
bigger gains would require fewer/larger images, not a different kernel.

Cross-page layout batching (the clean uniform-size win) was subsequently implemented —
see "Already done" above. It confirmed the model: **generation** batches well (memory-
bandwidth bound), **vision** does not (compute-bound).

## Plan — ranked by ROI

### Tier 1 — biggest win, moderate effort

**1. Packed (batched) content vision via block-diagonal attention.**
Today each crop runs its own `forward_vision` (+ its own `mx::eval` sync) → 26
forwards. The Qwen2-VL ViT supports packing multiple images into one sequence with a
**`cu_seqlens` block-diagonal mask** (each image attends only within itself) — exactly
how transformers/mlx-vlm process multi-image batches. Concatenate all crop
`pixel_values` (variable patch counts are fine — they're just rows), build the
block-diagonal mask, run **one** ViT forward, then slice per image.
- *Mechanism:* MLX `scaled_dot_product_attention` with an explicit additive mask;
  one batched `quantized`/bf16 matmul per layer instead of 26.
- *Expected:* collapse content-vision (31%) toward the cost of a single larger
  forward — potentially ~2–3× on that slice (~20% of total wall on a.pdf).
- *Risk:* medium (mask construction, per-image slicing); verify ViT(packed) ==
  ViT(per-image) like the `vlm_batch` test.

**2. Keep vision embeds device-resident (kill the GPU↔host round-trip).**
`forward_vision` currently `eval`s to a `std::vector<float>` (host), then
`build_multimodal_embeds` re-uploads it as an `array`. That's a forced sync + two
copies per block (~29/page). Return an `mx::array` and thread it through generation.
- *Mechanism:* keep the lazy graph on-device; only sync at the argmax.
- *Expected:* removes ~29 syncs/page; compounds with #1 (lets vision forwards overlap).
- *Risk:* low–medium (API change; the parity tests call the `vector<float>` path —
  keep it as a thin wrapper).

### Tier 2 — solid win, low–moderate risk

**3. `mx::compile` the hot forwards.** Wrap the LLM decode step and the ViT block in
`mx::compile` to fuse elementwise ops (rope, residuals, norms, gating) and cut
per-op Metal kernel-launch overhead (24 layers × ~15 ops/token = hundreds of launches
per token). mlx-lm does this.
- *Expected:* 10–20% on the generation + vision compute; same numerics.
- *Risk:* low (functional, same math); needs static shapes per call (bucket by length).

**4. Pipeline layout-vision and content-prep.** While the layout LLM is decoding
(16–20%), the page is idle on the GPU between token syncs — overlap content-crop
preprocessing / vision prefill using `mx::async_eval`.
- *Expected:* hides part of content-vision behind layout-generate.
- *Risk:* medium (scheduling/ordering).

### Tier 3 — larger effort or experimental

**5. Continuous batching for content generation.** Replace fixed length-buckets with
a dynamic batch that evicts finished sequences and admits new ones, keeping the GPU
saturated despite length variance (one long block currently bounds its bucket).
- *Expected:* meaningful on generation-bound docs (demo1-style).
- *Risk:* high (scheduler complexity).

**6. 8-bit vision quant experiment.** 4-bit vision was slower (compute-bound), but
8-bit may break even / win if the ViT attention is partly bandwidth-bound at high
patch counts. Measure layout-vision (1369 patches) specifically.
- *Risk:* low to try; revert if not faster.

**7. Custom Metal kernels — only if a hot spot survives.** After 1–4, re-profile. MLX's
built-in `sdpa` / `quantized_matmul` / `rms_norm` are already fused Metal kernels, so
hand-written kernels are likely unnecessary; reserve for a specific measured gap
(e.g., a fused rope+qkv or the patch-merger).

**8. Adaptive layout resolution.** Layout vision is a fixed 1036×1036 (1369 patches)
per page. For text-only pages a smaller raster would cut layout-vision (25%) with
little accuracy loss — gate on page content/DPI.

## Suggested order

1 → 2 (vision; the 55% on vision-bound docs) → 3 (compile; helps everything) →
re-profile → 5 (generation-bound docs) → 6/7/8 as measured.

Each step ships behind a parity check (packed==per-image, batched==sequential) and is
measured with the built-in profiler before/after.

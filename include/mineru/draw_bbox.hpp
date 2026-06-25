// Copyright (c) mlx-mineru.
// Layout / span visualization PDFs — the C++ analogue of MinerU draw_bbox.py. MinerU's gradio
// preview displays {name}_layout.pdf (translucent per-category boxes + red reading-order
// numbers). We reproduce it as a raster overlay: each rendered page + colored boxes baked in,
// wrapped back into a single PDF. Faithful to draw_layout_bbox / draw_span_bbox geometry &
// colors. The caller supplies pre-rendered pages (one per pdf_info page, same order).
#pragma once

#include <cstdint>
#include <vector>

#include "nlohmann/json.hpp"

namespace mineru {

struct RenderedPage {
  std::vector<uint8_t> rgb;  // w*h*3, top-down RGB8
  int w = 0, h = 0;
  double page_w_pt = 0.0, page_h_pt = 0.0;  // PDF page size in points (bbox space)
};

// {name}_layout.pdf: category fills (text/title/table/image/...) + sequential reading-order
// numbers, drawn from preproc_blocks (fallback para_blocks) + discarded_blocks.
std::vector<uint8_t> draw_layout_pdf(const nlohmann::json& pdf_info,
                                     const std::vector<RenderedPage>& pages, int jpeg_quality = 80);

// {name}_span.pdf: per-span stroked boxes (text/equation/image/table/dropped), no numbers.
std::vector<uint8_t> draw_span_pdf(const nlohmann::json& pdf_info,
                                   const std::vector<RenderedPage>& pages, int jpeg_quality = 80);

}  // namespace mineru

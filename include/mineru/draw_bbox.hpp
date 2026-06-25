// Copyright (c) mlx-mineru.
// Layout / span visualization PDFs — faithful C++ port of MinerU draw_bbox.py. Draws a vector
// overlay (translucent per-category boxes + red reading-order numbers) directly onto the
// original PDF pages via pdfium edit APIs, so the page content stays vector and the text under
// the boxes remains selectable — exactly like MinerU's reportlab+pypdf layout.pdf / span.pdf.
#pragma once

#include <cstdint>
#include <vector>

#include "nlohmann/json.hpp"

namespace mineru {

// {name}_layout.pdf: category fills (text/title/table/image/...) + sequential reading-order
// numbers, from preproc_blocks (fallback para_blocks) + discarded_blocks. pdf_bytes is the
// parsed PDF (same one the pdf_info was produced from).
std::vector<uint8_t> draw_layout_pdf(const nlohmann::json& pdf_info,
                                     const std::vector<uint8_t>& pdf_bytes);

// {name}_span.pdf: per-span stroked boxes (text/equation/image/table/dropped), no numbers.
std::vector<uint8_t> draw_span_pdf(const nlohmann::json& pdf_info,
                                   const std::vector<uint8_t>& pdf_bytes);

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Pipeline-backend multi-page driver: a per-page model_list + rendered page images ->
// middle_json pdf_info (assemble + post-OCR text-fill per page). Feed the result to
// union_make for Markdown / content_list. (Text/title path; visual spans + digital
// pdftext extraction are follow-ups.)
#pragma once

#include <cstdint>
#include <vector>

#include "mineru/layout_det.hpp"
#include "mineru/ocr_det.hpp"
#include "mineru/ocr_rec.hpp"
#include "nlohmann/json.hpp"

namespace mineru {

// Build one page's model_list entry (layout_dets + page_info) from scratch: PP-DocLayoutV2
// region boxes (with reading-order index) + OCR-det text-line boxes (label "ocr_text",
// empty text — filled later by post-OCR). rgb is the page image (w*h*3), boxes land in
// image-pixel space.
nlohmann::json build_page_model(const LayoutDetector& layout, const TextDetector& det,
                                const std::vector<uint8_t>& rgb, int w, int h);

struct PipelinePageImage {
  std::vector<uint8_t> rgb;  // rendered page, w*h*3 RGB8
  int w = 0, h = 0;          // image pixels
  int page_w = 0, page_h = 0;  // PDF page size in points (middle_json page_size)
};

// model_list[i] = {"layout_dets": [...], "page_info": {...}} for page i. Returns the
// pdf_info array (one assembled+filled page_info per page).
nlohmann::json pipeline_assemble_pages(const nlohmann::json& model_list,
                                       std::vector<PipelinePageImage>& pages,
                                       const TextRecognizer& rec);

}  // namespace mineru

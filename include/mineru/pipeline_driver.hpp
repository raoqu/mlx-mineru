// Copyright (c) mlx-mineru.
// Pipeline-backend multi-page driver: a per-page model_list + rendered page images ->
// middle_json pdf_info (assemble + post-OCR text-fill per page). Feed the result to
// union_make for Markdown / content_list. (Text/title path; visual spans + digital
// pdftext extraction are follow-ups.)
#pragma once

#include <cstdint>
#include <vector>

#include "mineru/formula_rec.hpp"
#include "mineru/layout_det.hpp"
#include "mineru/ocr.hpp"
#include "mineru/ocr_det.hpp"
#include "mineru/ocr_rec.hpp"
#include "mineru/pipeline_assemble.hpp"  // PageChar
#include "mineru/table_cls.hpp"
#include "mineru/table_rec.hpp"
#include "mineru/wired_table.hpp"
#include "nlohmann/json.hpp"

namespace mineru {

// Build one page's model_list entry (layout_dets + page_info) from scratch: PP-DocLayoutV2
// region boxes (with reading-order index) + OCR-det text-line boxes (label "ocr_text",
// empty text — filled later by post-OCR). rgb is the page image (w*h*3), boxes land in
// image-pixel space.
//   mfr       (optional): crop display/inline_formula regions -> attach `latex`.
//   ocr+table (optional, both required): crop table regions, OCR them, run SLANet+ ->
//             attach `html`.
//   table_cls + wired_rec (optional, both required): classify each table; for wired (or
//             low-confidence wireless) tables also run the UNet and pick the better HTML,
//             matching MinerU's wired/wireless selection.
// Per-page detection-stage timings in milliseconds, filled by build_page_model when a non-null
// `times` is passed (for the CLI/web per-page progress breakdown). All default 0.
struct PageStageTimes {
  double layout = 0;    // PP-DocLayoutV2 region detection
  double ocr_det = 0;   // DBNet text-line detection
  double table = 0;     // table OCR + SLANet+/UNet structure (sum over table regions)
  double formula = 0;   // MFR (sum over formula regions)
};

nlohmann::json build_page_model(const LayoutDetector& layout, const TextDetector& det,
                                const std::vector<uint8_t>& rgb, int w, int h,
                                const FormulaRecognizer* mfr = nullptr,
                                const OcrPipeline* ocr = nullptr,
                                const TableRecognizer* table_rec = nullptr,
                                const TableClassifier* table_cls = nullptr,
                                const WiredTableRecognizer* wired_rec = nullptr,
                                PageStageTimes* times = nullptr);

struct PipelinePageImage {
  std::vector<uint8_t> rgb;  // rendered page, w*h*3 RGB8
  int w = 0, h = 0;          // image pixels
  int page_w = 0, page_h = 0;  // PDF page size in points (middle_json page_size)
  std::vector<PageChar> chars;  // embedded PDF chars; non-empty -> digital text path
};

// Assemble + post-OCR text-fill for ONE page (no cross-page finalize): exactly one iteration
// of pipeline_assemble_pages. `page_idx` is the 0-based index within the document. Lets a caller
// process pages one at a time (detect -> recognize -> assemble) and stream per-page progress;
// call cross_page_table_merge() once over the full pdf_info afterwards.
nlohmann::json assemble_one_page(const nlohmann::json& model_page, PipelinePageImage& page,
                                 const TextRecognizer& rec, int page_idx);

// model_list[i] = {"layout_dets": [...], "page_info": {...}} for page i. Returns the
// pdf_info array (one assembled+filled page_info per page), with cross-page tables merged.
nlohmann::json pipeline_assemble_pages(const nlohmann::json& model_list,
                                       std::vector<PipelinePageImage>& pages,
                                       const TextRecognizer& rec);

}  // namespace mineru

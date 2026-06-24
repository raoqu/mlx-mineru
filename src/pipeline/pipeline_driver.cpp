// Copyright (c) mlx-mineru.
// Multi-page pipeline driver: assemble + post-OCR text-fill per page -> pdf_info.
#include "mineru/pipeline_driver.hpp"

#include <algorithm>

#include "mineru/pipeline_assemble.hpp"
#include "mineru/post_ocr.hpp"

namespace mineru {

nlohmann::json build_page_model(const LayoutDetector& layout, const TextDetector& det,
                                const std::vector<uint8_t>& rgb, int w, int h) {
  nlohmann::json dets = nlohmann::json::array();
  // Region boxes (reading-order index already assigned by the detector).
  for (const LayoutBox& b : layout.detect(rgb, w, h)) {
    dets.push_back({{"cls_id", b.cls_id}, {"label", b.label}, {"score", b.score},
                    {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}}, {"index", b.index}});
  }
  // OCR text-line boxes -> ocr_text dets (axis-aligned bbox, empty text).
  for (const DetBox& d : det.detect(rgb, w, h)) {
    int x0 = d.pts[0][0], y0 = d.pts[0][1], x1 = d.pts[0][0], y1 = d.pts[0][1];
    for (auto& p : d.pts) {
      x0 = std::min(x0, p[0]); y0 = std::min(y0, p[1]);
      x1 = std::max(x1, p[0]); y1 = std::max(y1, p[1]);
    }
    dets.push_back({{"label", "ocr_text"}, {"score", 1.0}, {"bbox", {x0, y0, x1, y1}},
                    {"text", ""}});
  }
  return {{"layout_dets", dets}, {"page_info", {{"width", w}, {"height", h}}}};
}

nlohmann::json pipeline_assemble_pages(const nlohmann::json& model_list,
                                       std::vector<PipelinePageImage>& pages,
                                       const TextRecognizer& rec) {
  nlohmann::json pdf_info = nlohmann::json::array();
  size_t n = std::min(model_list.size(), pages.size());
  for (size_t i = 0; i < n; ++i) {
    PipelinePageImage& pg = pages[i];
    nlohmann::json page_info = assemble_page_info(model_list[i], pg.page_w, pg.page_h, (int)i);
    double scale = pg.page_w > 0 ? (double)pg.w / pg.page_w : 1.0;
    fill_span_text(page_info, pg.rgb, pg.w, pg.h, scale, rec);
    pdf_info.push_back(std::move(page_info));
  }
  return pdf_info;
}

}  // namespace mineru

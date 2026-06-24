// Copyright (c) mlx-mineru.
// Multi-page pipeline driver: assemble + post-OCR text-fill per page -> pdf_info.
#include "mineru/pipeline_driver.hpp"

#include "mineru/pipeline_assemble.hpp"
#include "mineru/post_ocr.hpp"

namespace mineru {

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

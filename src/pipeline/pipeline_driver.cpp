// Copyright (c) mlx-mineru.
// Multi-page pipeline driver: assemble + post-OCR text-fill per page -> pdf_info.
#include "mineru/pipeline_driver.hpp"

#include <algorithm>
#include <array>

#include "mineru/pipeline_assemble.hpp"
#include "mineru/post_ocr.hpp"

namespace mineru {

// Crop an axis-aligned region [x0,y0,x1,y1] (image pixels) into a fresh RGB buffer.
static std::vector<uint8_t> crop_rgb(const std::vector<uint8_t>& rgb, int w, int h,
                                     const std::array<int, 4>& bb, int& cw, int& ch) {
  int x0 = std::max(0, std::min(w, bb[0])), y0 = std::max(0, std::min(h, bb[1]));
  int x1 = std::max(0, std::min(w, bb[2])), y1 = std::max(0, std::min(h, bb[3]));
  cw = std::max(0, x1 - x0);
  ch = std::max(0, y1 - y0);
  std::vector<uint8_t> c((size_t)cw * ch * 3);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      for (int k = 0; k < 3; ++k)
        c[((size_t)y * cw + x) * 3 + k] = rgb[((size_t)(y0 + y) * w + (x0 + x)) * 3 + k];
  return c;
}

nlohmann::json build_page_model(const LayoutDetector& layout, const TextDetector& det,
                                const std::vector<uint8_t>& rgb, int w, int h,
                                const FormulaRecognizer* mfr, const OcrPipeline* ocr,
                                const TableRecognizer* table_rec) {
  nlohmann::json dets = nlohmann::json::array();
  // Region boxes (reading-order index already assigned by the detector).
  for (const LayoutBox& b : layout.detect(rgb, w, h)) {
    nlohmann::json d = {{"cls_id", b.cls_id}, {"label", b.label}, {"score", b.score},
                        {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}}, {"index", b.index}};
    if (mfr && (b.label == "display_formula" || b.label == "inline_formula")) {
      int cw, ch;
      std::vector<uint8_t> crop = crop_rgb(rgb, w, h, b.bbox, cw, ch);
      d["latex"] = (cw > 0 && ch > 0) ? mfr->recognize(crop, cw, ch).latex : std::string();
    } else if (ocr && table_rec && b.label == "table") {
      int cw, ch;
      std::vector<uint8_t> crop = crop_rgb(rgb, w, h, b.bbox, cw, ch);
      std::string html;
      if (cw > 0 && ch > 0) {
        std::vector<TableOcrItem> items;
        for (const OcrLine& ln : ocr->run(crop, cw, ch))
          items.push_back({ln.box, ln.text, ln.score});
        html = table_rec->recognize_html(crop, cw, ch, items);
        // MinerU keeps only <table>..</table> (strips the <html><body> wrapper).
        size_t st = html.find("<table>"), en = html.rfind("</table>");
        if (st != std::string::npos && en != std::string::npos)
          html = html.substr(st, en + 8 - st);
      }
      d["html"] = html;
    }
    dets.push_back(std::move(d));
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
    // formula_number -> \tag{N} after the numbers are OCR-filled (both block lists, since
    // our assembly materializes para_blocks independently).
    optimize_formula_numbers(page_info["preproc_blocks"]);
    optimize_formula_numbers(page_info["para_blocks"]);
    pdf_info.push_back(std::move(page_info));
  }
  return pdf_info;
}

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Multi-page pipeline driver: assemble + post-OCR text-fill per page -> pdf_info.
#include "mineru/pipeline_driver.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "mineru/pipeline_assemble.hpp"
#include "mineru/post_ocr.hpp"
#include "mineru/table_merge.hpp"

namespace mineru {

// Keep only <table>..</table> (drop the SLANet/UNet <html><body> wrapper), matching MinerU.
static std::string strip_table_wrapper(std::string html) {
  size_t st = html.find("<table>"), en = html.rfind("</table>");
  if (st != std::string::npos && en != std::string::npos) html = html.substr(st, en + 8 - st);
  return html;
}

// --- Wired/wireless HTML selection heuristic (faithful to UnetTableModel.predict) ----------
// MinerU runs SLANet+ (wireless) on every table, then for tables the classifier marks wired
// (or wireless with low confidence) also runs the UNet (wired) model and picks the better of
// the two HTMLs by physical/non-blank cell counts and OCR-text coverage.

// Python3 round(): round-half-to-even. Inputs here are non-negative.
static long py_round(double x) {
  double f = std::floor(x), d = x - f;
  if (d < 0.5) return (long)f;
  if (d > 0.5) return (long)f + 1;
  long fi = (long)f;
  return (fi % 2 == 0) ? fi : fi + 1;
}

// count_table_cells_physical: non-overlapping "<td" + "<th" occurrences in the lowercased
// HTML (intentionally identical to MinerU, including the harmless <thead> over-count).
static int count_cells_physical(const std::string& html) {
  std::string l(html.size(), '\0');
  for (size_t i = 0; i < html.size(); ++i) l[i] = (char)std::tolower((unsigned char)html[i]);
  int n = 0;
  for (const char* sub : {"<td", "<th"}) {
    for (size_t p = l.find(sub); p != std::string::npos; p = l.find(sub, p + 3)) ++n;
  }
  return n;
}

// Inner text of each real <td>/<th> element, tags stripped and whitespace-trimmed (≈
// BeautifulSoup cell.text.strip()). Excludes <thead>/<tbody> (tag name must end at the cell).
static std::vector<std::string> cell_texts(const std::string& html) {
  std::vector<std::string> out;
  std::string l(html.size(), '\0');
  for (size_t i = 0; i < html.size(); ++i) l[i] = (char)std::tolower((unsigned char)html[i]);
  size_t p = 0;
  while (p < l.size()) {
    size_t lt = l.find('<', p);
    if (lt == std::string::npos) break;
    bool td = l.compare(lt, 3, "<td") == 0, th = l.compare(lt, 3, "<th") == 0;
    char after = lt + 3 < l.size() ? l[lt + 3] : '\0';
    if ((td || th) && (after == '>' || after == ' ' || after == '/' || after == '\t')) {
      size_t gt = l.find('>', lt);
      if (gt == std::string::npos) break;
      const char* close = td ? "</td" : "</th";
      size_t end = l.find(close, gt);
      std::string inner = (end == std::string::npos) ? html.substr(gt + 1)
                                                     : html.substr(gt + 1, end - gt - 1);
      // strip nested tags
      std::string txt;
      bool intag = false;
      for (char c : inner) {
        if (c == '<') intag = true;
        else if (c == '>') intag = false;
        else if (!intag) txt += c;
      }
      size_t b = txt.find_first_not_of(" \t\r\n");
      size_t e = txt.find_last_not_of(" \t\r\n");
      out.push_back(b == std::string::npos ? std::string() : txt.substr(b, e - b + 1));
      p = (end == std::string::npos) ? l.size() : end + 4;
    } else {
      p = lt + 1;
    }
  }
  return out;
}

// Return the chosen HTML between wired (UNet) and wireless (SLANet+). Falls back to wireless
// under the same conditions as MinerU's UnetTableModel.predict.
static std::string choose_table_html(const std::string& wired, const std::string& wireless,
                                     const std::vector<TableOcrItem>& items) {
  if (wired.empty()) return wireless;
  int wired_len = count_cells_physical(wired), wireless_len = count_cells_physical(wireless);
  int gap_of_len = wireless_len - wired_len;
  int wired_text = 0, wireless_text = 0;
  for (const TableOcrItem& it : items) {
    if (!it.text.empty() && wired.find(it.text) != std::string::npos) ++wired_text;
    if (!it.text.empty() && wireless.find(it.text) != std::string::npos) ++wireless_text;
  }
  auto blanks = [](const std::vector<std::string>& cs) {
    int b = 0;
    for (const std::string& c : cs) if (c.empty()) ++b;
    return b;
  };
  std::vector<std::string> wired_cells = cell_texts(wired), wireless_cells = cell_texts(wireless);
  int wired_non_blank = (int)wired_cells.size() - blanks(wired_cells);
  int wireless_non_blank = (int)wireless_cells.size() - blanks(wireless_cells);
  bool switch_flag = false;
  if (wireless_non_blank > wired_non_blank) {
    long scale = py_round(std::sqrt((double)wired_non_blank));
    long plus_2_cols = wired_non_blank + scale * 2;
    long sq_plus_2_rows = scale * (scale + 2);
    if ((long)(wireless_non_blank + 3) >= std::max(plus_2_cols, sq_plus_2_rows)) switch_flag = true;
  }
  bool to_wireless =
      switch_flag ||
      (gap_of_len >= 0 && gap_of_len <= 5 && wired_len <= py_round(wireless_len * 0.75)) ||
      (gap_of_len == 0 && wired_len <= 4) ||
      (wired_text <= wireless_text * 0.6 && wireless_text >= 10);
  return to_wireless ? wireless : wired;
}

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
                                const TableRecognizer* table_rec,
                                const TableClassifier* table_cls,
                                const WiredTableRecognizer* wired_rec, PageStageTimes* times) {
  using Clk = std::chrono::steady_clock;
  auto ms = [](Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  nlohmann::json dets = nlohmann::json::array();
  // Region boxes (reading-order index already assigned by the detector).
  auto _t0 = Clk::now();
  std::vector<LayoutBox> boxes = layout.detect(rgb, w, h);
  if (times) times->layout += ms(_t0, Clk::now());
  for (const LayoutBox& b : boxes) {
    nlohmann::json d = {{"cls_id", b.cls_id}, {"label", b.label}, {"score", b.score},
                        {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}}, {"index", b.index}};
    if (mfr && (b.label == "display_formula" || b.label == "inline_formula")) {
      auto _tf = Clk::now();
      int cw, ch;
      std::vector<uint8_t> crop = crop_rgb(rgb, w, h, b.bbox, cw, ch);
      d["latex"] = (cw > 0 && ch > 0) ? mfr->recognize(crop, cw, ch).latex : std::string();
      if (times) times->formula += ms(_tf, Clk::now());
    } else if (ocr && table_rec && b.label == "table") {
      auto _tt = Clk::now();
      int cw, ch;
      std::vector<uint8_t> crop = crop_rgb(rgb, w, h, b.bbox, cw, ch);
      // MINERU_DEBUG_TABLE=1: split a table's cost into OCR(Metal) / SLANet(ORT) / cls(Metal) /
      // UNet(Metal) — to see how much is the un-Metal-able SLANet vs the already-Metal parts.
      const bool dbg_tbl = std::getenv("MINERU_DEBUG_TABLE") != nullptr;
      double t_ocr = 0, t_slanet = 0, t_cls = 0, t_unet = 0;
      std::string html;
      if (cw > 0 && ch > 0) {
        std::vector<TableOcrItem> items;
        auto _o = Clk::now();
        for (const OcrLine& ln : ocr->run(crop, cw, ch))
          items.push_back({ln.box, ln.text, ln.score});
        t_ocr = ms(_o, Clk::now());
        // Always run SLANet+ (wireless) first; MinerU sets this as the table's HTML.
        auto _s = Clk::now();
        std::string wireless = strip_table_wrapper(table_rec->recognize_html(crop, cw, ch, items));
        t_slanet = ms(_s, Clk::now());
        html = wireless;
        // Route to the UNet (wired) model when the classifier says wired, or wireless with low
        // confidence (<0.9) — then pick the better of wired/wireless. Faithful to batch_analyze.
        if (table_cls && wired_rec && !items.empty()) {
          auto _c = Clk::now();
          TableClsResult cls = table_cls->classify(crop, cw, ch);
          t_cls = ms(_c, Clk::now());
          bool wired_candidate = cls.label == "wired_table" ||
                                 (cls.label == "wireless_table" && cls.score < 0.9f);
          if (wired_candidate) {
            auto _u = Clk::now();
            std::string wired = strip_table_wrapper(wired_rec->recognize_html(crop, cw, ch, items));
            t_unet = ms(_u, Clk::now());
            html = choose_table_html(wired, wireless, items);
          }
        }
      }
      if (dbg_tbl)
        std::fprintf(stderr, "[table-debug] OCR(Metal)=%.0f SLANet(ORT)=%.0f cls=%.0f UNet(Metal)=%.0f ms\n",
                     t_ocr, t_slanet, t_cls, t_unet);
      d["html"] = html;
      if (times) times->table += ms(_tt, Clk::now());
    }
    dets.push_back(std::move(d));
  }
  // OCR text-line boxes -> ocr_text dets (axis-aligned bbox, empty text).
  auto _td = Clk::now();
  std::vector<DetBox> det_boxes = det.detect(rgb, w, h);
  if (times) times->ocr_det += ms(_td, Clk::now());
  for (const DetBox& d : det_boxes) {
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

nlohmann::json assemble_one_page(const nlohmann::json& model_page, PipelinePageImage& pg,
                                 const TextRecognizer& rec, int page_idx) {
  nlohmann::json page_info = assemble_page_info(model_page, pg.page_w, pg.page_h, page_idx);
  double scale = pg.page_w > 0 ? (double)pg.w / pg.page_w : 1.0;
  // Digital PDF (embedded text) -> char-fill; scanned page -> OCR. For digital pages, any
  // span left empty by char-fill (e.g. text inside an image) falls back to OCR.
  if (!pg.chars.empty()) {
    fill_chars_in_page(page_info, pg.chars);
    fill_span_text(page_info, pg.rgb, pg.w, pg.h, scale, rec, /*min_confidence=*/0.5f,
                   /*only_empty=*/true);
  } else {
    fill_span_text(page_info, pg.rgb, pg.w, pg.h, scale, rec);
  }
  // formula_number -> \tag{N} after the numbers are OCR-filled (both block lists, since
  // our assembly materializes para_blocks independently).
  optimize_formula_numbers(page_info["preproc_blocks"]);
  optimize_formula_numbers(page_info["para_blocks"]);
  return page_info;
}

nlohmann::json pipeline_assemble_pages(const nlohmann::json& model_list,
                                       std::vector<PipelinePageImage>& pages,
                                       const TextRecognizer& rec) {
  nlohmann::json pdf_info = nlohmann::json::array();
  size_t n = std::min(model_list.size(), pages.size());
  for (size_t i = 0; i < n; ++i)
    pdf_info.push_back(assemble_one_page(model_list[i], pages[i], rec, (int)i));
  // Document-level finalize: merge tables that continue across a page break (faithful to
  // finalize_middle_json_from_preproc -> cross_page_table_merge). Title leveling stays a
  // no-op without the optional LLM config, matching MinerU's deterministic path.
  cross_page_table_merge(pdf_info);
  return pdf_info;
}

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Post-OCR text fill: crop each span from the page image and run batched OCR rec,
// faithful to MinerU _apply_post_ocr (rec-only, drop below min_confidence).
#include "mineru/post_ocr.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "mineru/ocr.hpp"  // get_rotate_crop_image (Quad)

namespace mineru {
namespace {
using nlohmann::json;
constexpr int kRecBatch = 6;

// A reference to one span plus its crop, so we can write results back in place.
struct SpanJob {
  json* span;
  std::vector<uint8_t> crop;
  int w, h;
};

void collect(json& blocks, const std::vector<uint8_t>& rgb, int W, int H, double scale,
             std::vector<SpanJob>& jobs, bool only_empty) {
  for (auto& b : blocks) {
    if (!b.contains("lines")) continue;
    for (auto& ln : b["lines"]) {
      if (!ln.contains("spans")) continue;
      for (auto& sp : ln["spans"]) {
        if (sp.value("type", "") != "text") continue;
        if (only_empty && !sp.value("content", "").empty()) continue;
        auto bb = sp["bbox"];
        // span bbox is page-point; scale up to image coords, build a 4-corner quad.
        float x0 = bb[0].get<float>() * scale, y0 = bb[1].get<float>() * scale;
        float x1 = bb[2].get<float>() * scale, y1 = bb[3].get<float>() * scale;
        Quad q{{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
        int cw, ch;
        auto crop = get_rotate_crop_image(rgb, W, H, q, cw, ch);
        jobs.push_back({&sp, std::move(crop), cw, ch});
      }
    }
  }
}
}  // namespace

void fill_span_text(json& page_info, const std::vector<uint8_t>& rgb, int W, int H, double scale,
                    const TextRecognizer& rec, float min_confidence, bool only_empty) {
  // MinerU runs post-OCR on preproc/discarded then para_split copies the filled spans into
  // para_blocks. Our assembly already materialized para_blocks (independent json copies), so
  // fill all three to keep them consistent for the downstream union_make.
  std::vector<SpanJob> jobs;
  if (page_info.contains("preproc_blocks")) collect(page_info["preproc_blocks"], rgb, W, H, scale, jobs, only_empty);
  if (page_info.contains("discarded_blocks")) collect(page_info["discarded_blocks"], rgb, W, H, scale, jobs, only_empty);
  if (page_info.contains("para_blocks")) collect(page_info["para_blocks"], rgb, W, H, scale, jobs, only_empty);
  if (jobs.empty()) return;

  // Batched rec (predict_rec semantics): sort by aspect, batches of 6 share the widest
  // crop's max_wh_ratio.
  std::vector<double> aspect(jobs.size());
  for (size_t i = 0; i < jobs.size(); ++i)
    aspect[i] = jobs[i].h > 0 ? (double)jobs[i].w / jobs[i].h : 0.0;
  std::vector<int> idx(jobs.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return aspect[a] < aspect[b]; });

  for (size_t beg = 0; beg < idx.size(); beg += kRecBatch) {
    size_t end = std::min(idx.size(), beg + kRecBatch);
    double max_wh = aspect[idx[end - 1]];
    for (size_t k = beg; k < end; ++k) {
      SpanJob& j = jobs[idx[k]];
      if (j.w <= 0 || j.h <= 0) { (*j.span)["content"] = ""; (*j.span)["score"] = 0.0; continue; }
      RecResult r = rec.recognize(j.crop, j.w, j.h, max_wh);
      if (r.score > min_confidence) {
        (*j.span)["content"] = r.text;
        (*j.span)["score"] = std::round(r.score * 1000.0) / 1000.0;  // round(score, 3)
      } else {
        (*j.span)["content"] = "";
        (*j.span)["score"] = 0.0;
      }
    }
  }
}

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Raster-overlay layout/span PDFs (see header). Geometry & colors faithful to MinerU
// draw_bbox.py; rendering is a raster blend (alpha 0.3 fills, 1px strokes) rather than a
// vector overlay, which is what the preview needs and keeps us off pdfium's edit APIs.
#include "mineru/draw_bbox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "mineru/image_write.hpp"

namespace mineru {
namespace {

using json = nlohmann::json;

struct RGB { int r, g, b; };

// 5x7 bitmap font, digits 0-9 (msb = leftmost of 5 columns).
constexpr uint8_t kFont[10][7] = {
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},  // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},  // 1
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},  // 2
    {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},  // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},  // 4
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},  // 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},  // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},  // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},  // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},  // 9
};

struct Canvas {
  std::vector<uint8_t> px;  // mutable copy of rgb
  int w, h;
  void blend(int x, int y, RGB c, double a) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    uint8_t* p = &px[((size_t)y * w + x) * 3];
    p[0] = (uint8_t)std::lround(c.r * a + p[0] * (1 - a));
    p[1] = (uint8_t)std::lround(c.g * a + p[1] * (1 - a));
    p[2] = (uint8_t)std::lround(c.b * a + p[2] * (1 - a));
  }
  void fill_rect(int x0, int y0, int x1, int y1, RGB c, double a) {
    x0 = std::max(0, x0); y0 = std::max(0, y0);
    x1 = std::min(w, x1); y1 = std::min(h, y1);
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) blend(x, y, c, a);
  }
  void stroke_rect(int x0, int y0, int x1, int y1, RGB c, int t) {
    for (int k = 0; k < t; ++k) {
      for (int x = x0; x <= x1; ++x) { blend(x, y0 + k, c, 1.0); blend(x, y1 - k, c, 1.0); }
      for (int y = y0; y <= y1; ++y) { blend(x0 + k, y, c, 1.0); blend(x1 - k, y, c, 1.0); }
    }
  }
  void glyph(int gx, int gy, int digit, int s, RGB c) {
    for (int row = 0; row < 7; ++row)
      for (int col = 0; col < 5; ++col)
        if (kFont[digit][row] & (1 << (4 - col)))
          fill_rect(gx + col * s, gy + row * s, gx + col * s + s, gy + row * s + s, c, 1.0);
  }
  void number(int x, int y, int n, int s, RGB c) {
    std::string str = std::to_string(n);
    for (char ch : str) {
      glyph(x, y, ch - '0', s, c);
      x += 6 * s;  // 5 cols + 1 gap
    }
  }
};

// Build a single PDF embedding one JPEG-image page per canvas at its point size.
std::vector<uint8_t> pages_to_pdf(const std::vector<std::vector<uint8_t>>& jpegs,
                                  const std::vector<std::array<double, 2>>& sizes_pt,
                                  const std::vector<std::array<int, 2>>& px) {
  auto num = [](double v) {
    char b[48];
    std::snprintf(b, sizeof(b), "%.4f", v);
    return std::string(b);
  };
  std::string head = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";
  std::vector<uint8_t> out(head.begin(), head.end());
  auto app = [&](const std::string& s) { out.insert(out.end(), s.begin(), s.end()); };
  int n = (int)jpegs.size();
  std::vector<size_t> off;
  int obj = 1;
  auto begin = [&](int id) { off.resize(std::max((int)off.size(), id + 1)); off[id] = out.size();
                             app(std::to_string(id) + " 0 obj\n"); };
  // 1 catalog, 2 pages, then per page: page obj, image obj, content obj.
  begin(1); app("<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  std::string kids;
  for (int i = 0; i < n; ++i) kids += std::to_string(3 + i * 3) + " 0 R ";
  begin(2);
  app("<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(n) + " >>\nendobj\n");
  obj = 3;
  for (int i = 0; i < n; ++i) {
    int page_id = obj, img_id = obj + 1, cont_id = obj + 2;
    obj += 3;
    double W = sizes_pt[i][0], H = sizes_pt[i][1];
    begin(page_id);
    app("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + num(W) + " " + num(H) +
        "] /Resources << /XObject << /Im0 " + std::to_string(img_id) +
        " 0 R >> >> /Contents " + std::to_string(cont_id) + " 0 R >>\nendobj\n");
    begin(img_id);
    app("<< /Type /XObject /Subtype /Image /Width " + std::to_string(px[i][0]) + " /Height " +
        std::to_string(px[i][1]) +
        " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length " +
        std::to_string(jpegs[i].size()) + " >>\nstream\n");
    out.insert(out.end(), jpegs[i].begin(), jpegs[i].end());
    app("\nendstream\nendobj\n");
    begin(cont_id);
    std::string c = "q " + num(W) + " 0 0 " + num(H) + " 0 0 cm /Im0 Do Q\n";
    app("<< /Length " + std::to_string(c.size()) + " >>\nstream\n" + c + "endstream\nendobj\n");
  }
  size_t xref = out.size();
  int total = obj;  // ids 1..total-1
  app("xref\n0 " + std::to_string(total) + "\n0000000000 65535 f \n");
  for (int i = 1; i < total; ++i) {
    char e[32];
    std::snprintf(e, sizeof(e), "%010zu 00000 n \n", off[i]);
    app(e);
  }
  app("trailer\n<< /Size " + std::to_string(total) + " /Root 1 0 R >>\nstartxref\n" +
      std::to_string(xref) + "\n%%EOF\n");
  return out;
}

const json& source_blocks(const json& page) {
  static const json empty = json::array();
  if (page.contains("preproc_blocks") && !page["preproc_blocks"].empty()) return page["preproc_blocks"];
  if (page.contains("para_blocks")) return page["para_blocks"];
  return empty;
}

bool is_text_like(const std::string& t) {
  return t == "text" || t == "ref_text" || t == "abstract" || t == "phonetic";
}
bool is_direct_layout(const std::string& t) {
  return is_text_like(t) || t == "title" || t == "interline_equation" || t == "list" || t == "index";
}

std::array<int, 4> scaled(const json& bbox, double sx, double sy) {
  return {(int)std::lround(bbox[0].get<double>() * sx), (int)std::lround(bbox[1].get<double>() * sy),
          (int)std::lround(bbox[2].get<double>() * sx), (int)std::lround(bbox[3].get<double>() * sy)};
}

}  // namespace

std::vector<uint8_t> draw_layout_pdf(const json& pdf_info, const std::vector<RenderedPage>& pages,
                                     int jpeg_quality) {
  std::vector<std::vector<uint8_t>> jpegs;
  std::vector<std::array<double, 2>> sizes;
  std::vector<std::array<int, 2>> px;
  size_t n = std::min(pdf_info.size(), pages.size());
  for (size_t i = 0; i < n; ++i) {
    const json& page = pdf_info[i];
    const RenderedPage& rp = pages[i];
    double pw = page.contains("page_size") ? page["page_size"][0].get<double>() : rp.page_w_pt;
    double ph = page.contains("page_size") ? page["page_size"][1].get<double>() : rp.page_h_pt;
    double sx = pw > 0 ? rp.w / pw : 1.0, sy = ph > 0 ? rp.h / ph : 1.0;
    Canvas cv{rp.rgb, rp.w, rp.h};

    // Category fills (alpha 0.3) in MinerU's draw order; list_items are stroked.
    struct Cat { std::vector<std::array<int, 4>> rects; RGB c; bool fill; };
    std::vector<Cat> cats;
    cats.reserve(16);  // keep references from add() stable (no reallocation)
    auto add = [&](RGB c, bool fill) -> std::vector<std::array<int, 4>>& {
      cats.push_back({{}, c, fill});
      return cats.back().rects;
    };
    auto& tables_body = add({204, 204, 0}, true);
    auto& tables_cap = add({255, 255, 102}, true);
    auto& tables_foot = add({229, 255, 204}, true);
    auto& imgs_body = add({153, 255, 51}, true);
    auto& imgs_cap = add({102, 178, 255}, true);
    auto& imgs_foot = add({255, 178, 102}, true);
    auto& titles = add({102, 102, 255}, true);
    auto& texts = add({153, 0, 76}, true);
    auto& eqs = add({0, 255, 0}, true);
    auto& lists = add({40, 169, 92}, true);
    auto& list_items = add({40, 169, 92}, false);
    auto& indices = add({40, 169, 92}, true);
    auto& dropped = add({158, 158, 158}, true);

    for (const json& b : page.value("discarded_blocks", json::array()))
      if (b.contains("bbox")) dropped.push_back(scaled(b["bbox"], sx, sy));

    std::vector<std::array<int, 4>> numbered;  // layout_bbox_list, in reading order
    for (const json& block : source_blocks(page)) {
      std::string t = block.value("type", "");
      if (t == "table" || t == "image" || t == "chart" || t == "code") {
        for (const json& sub : block.value("blocks", json::array())) {
          std::string st = sub.value("type", "");
          if (!sub.contains("bbox")) continue;
          if (sub.value("cross_page", false)) continue;
          auto r = scaled(sub["bbox"], sx, sy);
          if (st == "table_body") tables_body.push_back(r);
          else if (st == "table_caption") tables_cap.push_back(r);
          else if (st == "table_footnote") tables_foot.push_back(r);
          else if (st == "image_body" || st == "chart_body") imgs_body.push_back(r);
          else if (st == "image_caption" || st == "chart_caption") imgs_cap.push_back(r);
          else if (st == "image_footnote" || st == "chart_footnote") imgs_foot.push_back(r);
          numbered.push_back(r);
        }
      } else if (block.contains("bbox")) {
        auto r = scaled(block["bbox"], sx, sy);
        if (t == "title") titles.push_back(r);
        else if (is_text_like(t)) texts.push_back(r);
        else if (t == "interline_equation") eqs.push_back(r);
        else if (t == "list") { lists.push_back(r);
          for (const json& sub : block.value("blocks", json::array()))
            if (sub.contains("bbox")) list_items.push_back(scaled(sub["bbox"], sx, sy)); }
        else if (t == "index") indices.push_back(r);
        if (is_direct_layout(t)) numbered.push_back(r);
      }
    }

    for (const Cat& cat : cats)
      for (const auto& r : cat.rects) {
        if (cat.fill) cv.fill_rect(r[0], r[1], r[2], r[3], cat.c, 0.3);
        else cv.stroke_rect(r[0], r[1], r[2], r[3], cat.c, 1);
      }
    // Sequential reading-order numbers (red), top-right of each layout block.
    int fontpx = std::max(7, (int)std::lround(10.0 * sy));
    int s = std::max(1, fontpx / 7);
    for (size_t k = 0; k < numbered.size(); ++k) {
      const auto& r = numbered[k];
      int nx = std::min(cv.w - 6 * s * 2, r[2] + 2);
      cv.number(nx, std::max(0, r[1]), (int)k + 1, s, {255, 0, 0});
    }

    jpegs.push_back(encode_jpeg(cv.px, rp.w, rp.h, jpeg_quality));
    sizes.push_back({pw, ph});
    px.push_back({rp.w, rp.h});
  }
  return pages_to_pdf(jpegs, sizes, px);
}

std::vector<uint8_t> draw_span_pdf(const json& pdf_info, const std::vector<RenderedPage>& pages,
                                   int jpeg_quality) {
  std::vector<std::vector<uint8_t>> jpegs;
  std::vector<std::array<double, 2>> sizes;
  std::vector<std::array<int, 2>> px;
  size_t n = std::min(pdf_info.size(), pages.size());
  for (size_t i = 0; i < n; ++i) {
    const json& page = pdf_info[i];
    const RenderedPage& rp = pages[i];
    double pw = page.contains("page_size") ? page["page_size"][0].get<double>() : rp.page_w_pt;
    double ph = page.contains("page_size") ? page["page_size"][1].get<double>() : rp.page_h_pt;
    double sx = pw > 0 ? rp.w / pw : 1.0, sy = ph > 0 ? rp.h / ph : 1.0;
    Canvas cv{rp.rgb, rp.w, rp.h};

    auto draw_span = [&](const json& span, RGB c) {
      if (span.contains("bbox")) { auto r = scaled(span["bbox"], sx, sy); cv.stroke_rect(r[0], r[1], r[2], r[3], c, 1); }
    };
    auto walk = [&](const json& block) {
      for (const json& line : block.value("lines", json::array()))
        for (const json& span : line.value("spans", json::array())) {
          std::string st = span.value("type", "");
          if (st == "text") draw_span(span, {255, 0, 0});
          else if (st == "inline_equation") draw_span(span, {0, 255, 0});
          else if (st == "interline_equation") draw_span(span, {0, 0, 255});
          else if (st == "image" || st == "chart") draw_span(span, {255, 204, 0});
          else if (st == "table") draw_span(span, {204, 0, 255});
        }
    };
    for (const json& b : page.value("discarded_blocks", json::array()))
      for (const json& line : b.value("lines", json::array()))
        for (const json& span : line.value("spans", json::array()))
          draw_span(span, {158, 158, 158});
    for (const json& block : source_blocks(page)) {
      std::string t = block.value("type", "");
      if (t == "table" || t == "image" || t == "chart" || t == "code")
        for (const json& sub : block.value("blocks", json::array())) walk(sub);
      else
        walk(block);
    }
    jpegs.push_back(encode_jpeg(cv.px, rp.w, rp.h, jpeg_quality));
    sizes.push_back({pw, ph});
    px.push_back({rp.w, rp.h});
  }
  return pages_to_pdf(jpegs, sizes, px);
}

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Vector-overlay layout/span PDFs (see header). Faithful to MinerU draw_bbox.py: same block
// collection, colors, draw order, 0.3 fill alpha, 1pt strokes, and reading-order numbers
// (Helvetica 10, top-right). Implemented with pdfium edit APIs (rect/text page objects +
// SaveAsCopy) instead of reportlab+pypdf, preserving the original page content.
#include "mineru/draw_bbox.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fpdf_edit.h"
#include "fpdf_save.h"
#include "fpdfview.h"
#include "mineru/pdf.hpp"

namespace mineru {
namespace {

using json = nlohmann::json;

struct RGB { int r, g, b; };

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

// Collect bytes from FPDF_SaveAsCopy.
struct ByteWriter {
  FPDF_FILEWRITE fw;
  std::vector<uint8_t>* out;
};
int write_block(FPDF_FILEWRITE* pThis, const void* data, unsigned long size) {
  auto* w = reinterpret_cast<ByteWriter*>(pThis);
  const uint8_t* p = static_cast<const uint8_t*>(data);
  w->out->insert(w->out->end(), p, p + size);
  return 1;
}
std::vector<uint8_t> save_doc(FPDF_DOCUMENT doc) {
  std::vector<uint8_t> out;
  ByteWriter w{};
  w.fw.version = 1;
  w.fw.WriteBlock = write_block;
  w.out = &out;
  FPDF_SaveAsCopy(doc, &w.fw, FPDF_NO_INCREMENTAL);
  return out;
}

// MinerU bbox is top-left point space; pdfium user space is bottom-left. For rotation 0 (all
// our inputs) the rect is [x0, page_h - y1, w, h]; numbers anchor at (x1+2, page_h - y0 - 10).
void add_rect(FPDF_PAGE page, const json& bbox, double page_h, RGB c, bool fill) {
  double x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
  FPDF_PAGEOBJECT r = FPDFPageObj_CreateNewRect((float)x0, (float)(page_h - y1), (float)(x1 - x0),
                                                (float)(y1 - y0));
  if (fill) {
    // Highlight: a SOLID color pre-blended as 0.3-over-white, inserted BENEATH the page content
    // (index 0) so the text renders on top — the same look as a 0.3 alpha fill, but with NO
    // per-object alpha. We previously used FPDFPageObj_SetFillColor(...,77); pdfium emits that
    // alpha as a named ExtGState added to the page, and on PDFs whose pages already define
    // ExtGStates it writes a content stream referencing a name (e.g. /FXE4) that it never
    // registers in the page's /ExtGState dict — so viewers drop the alpha and the fill renders
    // OPAQUE, hiding the text. A solid color needs no ExtGState, so it is reliable in every
    // viewer and every source PDF; drawing it underneath keeps the text legible.
    auto blend = [](unsigned int v) -> unsigned int {
      return (unsigned int)(0.3 * v + 0.7 * 255 + 0.5);
    };
    FPDFPageObj_SetFillColor(r, blend(c.r), blend(c.g), blend(c.b), 255);
    FPDFPath_SetDrawMode(r, FPDF_FILLMODE_WINDING, 0);
    FPDFPage_InsertObjectAtIndex(page, r, 0);  // beneath the original page content
  } else {
    FPDFPageObj_SetStrokeColor(r, c.r, c.g, c.b, 255);
    FPDFPageObj_SetStrokeWidth(r, 1.0f);
    FPDFPath_SetDrawMode(r, 0, 1);
    FPDFPage_InsertObject(page, r);  // outline on top
  }
}

void add_number(FPDF_DOCUMENT doc, FPDF_PAGE page, const json& bbox, double page_h, int n) {
  double x1 = bbox[2], y0 = bbox[1];
  FPDF_PAGEOBJECT t = FPDFPageObj_NewTextObj(doc, "Helvetica", 10.0f);
  std::string s = std::to_string(n);
  std::vector<unsigned short> u16(s.begin(), s.end());
  u16.push_back(0);
  FPDFText_SetText(t, u16.data());
  FPDFPageObj_SetFillColor(t, 255, 0, 0, 255);
  FS_MATRIX m{1, 0, 0, 1, (float)(x1 + 2), (float)(page_h - y0 - 10)};
  FPDFPageObj_SetMatrix(t, &m);
  FPDFPage_InsertObject(page, t);
}

}  // namespace

std::vector<uint8_t> draw_layout_pdf(const json& pdf_info, const std::vector<uint8_t>& pdf_bytes) {
  pdfium_acquire();
  FPDF_DOCUMENT doc = FPDF_LoadMemDocument(pdf_bytes.data(), (int)pdf_bytes.size(), nullptr);
  if (!doc) { pdfium_release(); return {}; }

  for (size_t i = 0; i < pdf_info.size(); ++i) {
    const json& pinfo = pdf_info[i];
    int page_idx = pinfo.value("page_idx", (int)i);
    FPDF_PAGE page = FPDF_LoadPage(doc, page_idx);
    if (!page) continue;
    double page_h = FPDF_GetPageHeightF(page);

    // Per-category fills in MinerU's draw order; list_items are stroked. (r,g,b,fill)
    struct Item { json bbox; RGB c; bool fill; };
    std::vector<Item> items;          // category boxes, drawn first
    std::vector<json> numbered;       // layout_bbox_list, reading order
    auto push_cat = [&](const json& b, RGB c, bool fill) { items.push_back({b, c, fill}); };

    // discarded first collected, but MinerU draws codes/caption/footnote/dropped early; we honor
    // its exact ordering below by category buckets.
    std::vector<json> dropped, tb, tcap, tfoot, ib, icap, ifoot, titles, texts, eqs, lists, litems, idx;
    for (const json& b : pinfo.value("discarded_blocks", json::array()))
      if (b.contains("bbox")) dropped.push_back(b["bbox"]);

    for (const json& block : source_blocks(pinfo)) {
      std::string t = block.value("type", "");
      if (t == "table" || t == "image" || t == "chart" || t == "code") {
        for (const json& sub : block.value("blocks", json::array())) {
          if (!sub.contains("bbox") || sub.value("cross_page", false)) continue;
          std::string st = sub.value("type", "");
          const json& bb = sub["bbox"];
          if (st == "table_body") tb.push_back(bb);
          else if (st == "table_caption") tcap.push_back(bb);
          else if (st == "table_footnote") tfoot.push_back(bb);
          else if (st == "image_body" || st == "chart_body") ib.push_back(bb);
          else if (st == "image_caption" || st == "chart_caption") icap.push_back(bb);
          else if (st == "image_footnote" || st == "chart_footnote") ifoot.push_back(bb);
          numbered.push_back(bb);
        }
      } else if (block.contains("bbox")) {
        const json& bb = block["bbox"];
        if (t == "title") titles.push_back(bb);
        else if (is_text_like(t)) texts.push_back(bb);
        else if (t == "interline_equation") eqs.push_back(bb);
        else if (t == "list") {
          lists.push_back(bb);
          for (const json& sub : block.value("blocks", json::array()))
            if (sub.contains("bbox")) litems.push_back(sub["bbox"]);
        } else if (t == "index") idx.push_back(bb);
        if (is_direct_layout(t)) numbered.push_back(bb);
      }
    }
    // Draw order faithful to draw_layout_bbox (codes omitted — pipeline has none here).
    // Text-ish regions get a translucent highlight (drawn beneath the text by add_rect); the
    // image/chart/table BODIES are outlined on top instead — a behind-highlight would be hidden
    // by the opaque figure/image anyway, so an outline marks them without obscuring the content.
    for (auto& b : dropped) push_cat(b, {158, 158, 158}, true);
    for (auto& b : tb) push_cat(b, {204, 204, 0}, /*fill=*/false);
    for (auto& b : tcap) push_cat(b, {255, 255, 102}, true);
    for (auto& b : tfoot) push_cat(b, {229, 255, 204}, true);
    for (auto& b : ib) push_cat(b, {153, 255, 51}, /*fill=*/false);
    for (auto& b : icap) push_cat(b, {102, 178, 255}, true);
    for (auto& b : ifoot) push_cat(b, {255, 178, 102}, true);
    for (auto& b : titles) push_cat(b, {102, 102, 255}, true);
    for (auto& b : texts) push_cat(b, {153, 0, 76}, true);
    for (auto& b : eqs) push_cat(b, {0, 255, 0}, true);
    for (auto& b : lists) push_cat(b, {40, 169, 92}, true);
    for (auto& b : litems) push_cat(b, {40, 169, 92}, false);
    for (auto& b : idx) push_cat(b, {40, 169, 92}, true);

    // DEBUG (MINERU_DEBUG_BBOX=1): per page, report how many FILL rects are drawn and the maximum
    // overlap/stack depth (how many fills cover the most-covered point) — to settle whether an
    // "opaque" region comes from stacking many semi-transparent fills vs a single fill rendered
    // opaque. Dump every fill rect for the page given by MINERU_DEBUG_BBOX_PAGE (0-based).
    if (const char* dbg = std::getenv("MINERU_DEBUG_BBOX"); dbg && *dbg && *dbg != '0') {
      int dbg_page = 0;
      if (const char* dp = std::getenv("MINERU_DEBUG_BBOX_PAGE")) dbg_page = std::atoi(dp);
      int nfill = 0;
      std::vector<std::array<double, 4>> fills;
      for (const Item& it : items)
        if (it.fill) {
          ++nfill;
          fills.push_back({it.bbox[0], it.bbox[1], it.bbox[2], it.bbox[3]});
        }
      // max stack depth: sweep over all fill rects' corners as candidate points.
      int max_depth = 0;
      for (const auto& a : fills) {
        double cx = (a[0] + a[2]) / 2, cy = (a[1] + a[3]) / 2;  // center of each rect
        int d = 0;
        for (const auto& b : fills)
          if (b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3]) ++d;
        max_depth = std::max(max_depth, d);
      }
      std::fprintf(stderr,
                   "[bbox-debug] page_idx=%d: %d fill rect(s), %d stroke rect(s), max stack depth=%d\n",
                   page_idx, nfill, (int)items.size() - nfill, max_depth);
      if (page_idx == dbg_page) {
        for (const Item& it : items)
          if (it.fill)
            std::fprintf(stderr, "[bbox-debug]   fill rgb(%d,%d,%d) bbox[%.0f,%.0f,%.0f,%.0f]\n",
                         it.c.r, it.c.g, it.c.b, (double)it.bbox[0], (double)it.bbox[1],
                         (double)it.bbox[2], (double)it.bbox[3]);
      }
    }

    for (const Item& it : items) add_rect(page, it.bbox, page_h, it.c, it.fill);
    for (size_t k = 0; k < numbered.size(); ++k) add_number(doc, page, numbered[k], page_h, (int)k + 1);

    FPDFPage_GenerateContent(page);
    FPDF_ClosePage(page);
  }

  std::vector<uint8_t> out = save_doc(doc);
  FPDF_CloseDocument(doc);
  pdfium_release();
  return out;
}

std::vector<uint8_t> draw_span_pdf(const json& pdf_info, const std::vector<uint8_t>& pdf_bytes) {
  pdfium_acquire();
  FPDF_DOCUMENT doc = FPDF_LoadMemDocument(pdf_bytes.data(), (int)pdf_bytes.size(), nullptr);
  if (!doc) { pdfium_release(); return {}; }

  for (size_t i = 0; i < pdf_info.size(); ++i) {
    const json& pinfo = pdf_info[i];
    int page_idx = pinfo.value("page_idx", (int)i);
    FPDF_PAGE page = FPDF_LoadPage(doc, page_idx);
    if (!page) continue;
    double page_h = FPDF_GetPageHeightF(page);

    auto span_color = [](const std::string& st, RGB& c) -> bool {
      if (st == "text") { c = {255, 0, 0}; return true; }
      if (st == "inline_equation") { c = {0, 255, 0}; return true; }
      if (st == "interline_equation") { c = {0, 0, 255}; return true; }
      if (st == "image" || st == "chart") { c = {255, 204, 0}; return true; }
      if (st == "table") { c = {204, 0, 255}; return true; }
      return false;
    };
    auto walk = [&](const json& block) {
      for (const json& line : block.value("lines", json::array()))
        for (const json& span : line.value("spans", json::array())) {
          RGB c{};
          if (span.contains("bbox") && span_color(span.value("type", ""), c))
            add_rect(page, span["bbox"], page_h, c, /*fill=*/false);
        }
    };
    for (const json& b : pinfo.value("discarded_blocks", json::array()))
      for (const json& line : b.value("lines", json::array()))
        for (const json& span : line.value("spans", json::array()))
          if (span.contains("bbox")) add_rect(page, span["bbox"], page_h, {158, 158, 158}, false);
    for (const json& block : source_blocks(pinfo)) {
      std::string t = block.value("type", "");
      if (t == "table" || t == "image" || t == "chart" || t == "code")
        for (const json& sub : block.value("blocks", json::array())) walk(sub);
      else
        walk(block);
    }
    FPDFPage_GenerateContent(page);
    FPDF_ClosePage(page);
  }

  std::vector<uint8_t> out = save_doc(doc);
  FPDF_CloseDocument(doc);
  pdfium_release();
  return out;
}

}  // namespace mineru

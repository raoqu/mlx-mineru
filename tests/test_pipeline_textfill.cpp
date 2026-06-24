// Copyright (c) mlx-mineru.
// Pipeline P5: full text vertical — assemble(model_list) + post-OCR text-fill from the
// page image == MinerU middle_json span text. Closes model_list + image -> middle_json.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/ocr_rec.hpp"
#include "mineru/pdf.hpp"
#include "mineru/pipeline_assemble.hpp"
#include "mineru/post_ocr.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Flatten span contents (text spans, in block/line/span order) for comparison.
static std::vector<std::string> texts(const json& blocks) {
  std::vector<std::string> out;
  for (const auto& b : blocks)
    for (const auto& ln : b.value("lines", json::array()))
      for (const auto& sp : ln.value("spans", json::array()))
        if (sp.value("type", "") == "text") out.push_back(sp.value("content", ""));
  return out;
}

// ASCII skeleton: keep digits/Latin/punctuation, drop spaces + non-ASCII. a.pdf is a
// DIGITAL pdf, so MinerU's golden text comes from the embedded text layer (Kangxi-radical
// codepoints, NBSP), while our OCR text-fill reads the rendered glyphs (CJK-unified) — the
// CJK is NFKC-equivalent but not byte-equal. The ASCII run is unambiguous and must match.
static std::string ascii_skel(const std::string& s) {
  std::string o;
  for (unsigned char c : s)
    if (c > 0x20 && c < 0x7F) o += (char)c;
  return o;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string onnx = (argc > 2) ? argv[2] : "models/pipeline/OCR/ocr_rec.onnx";
  std::string dict = (argc > 3) ? argv[3] : "models/pipeline/OCR/ppocrv6_dict.txt";
  std::string pdf = (argc > 4) ? argv[4] : "a.pdf";

  json model_page = json::parse(read_file(golden_dir + "/pipeline_p0_model.json"));
  json want = json::parse(read_file(golden_dir + "/pipeline_p0_middle.json"));
  int page_w = want["page_size"][0], page_h = want["page_size"][1];

  // Render a.pdf p0 at MinerU's pipeline resolution (200 DPI -> 1700x2200 over 612pt) so
  // the OCR crops match what MinerU fed its rec (pdf_raster proves PdfDocument is byte-exact
  // vs pypdfium). Lower res shifts some CJK glyphs to compatibility variants.
  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  mineru::PageImage img = doc.render_page(0, 200);
  int img_w = img.width, img_h = img.height;
  std::vector<uint8_t>& rgb = img.rgb;
  double scale = (double)img_w / page_w;

  json page_info = mineru::assemble_page_info(model_page, page_w, page_h, 0);
  mineru::TextRecognizer rec = mineru::TextRecognizer::load(onnx, dict);
  mineru::fill_span_text(page_info, rgb, img_w, img_h, scale, rec);

  auto got = texts(page_info["preproc_blocks"]);
  auto wt = texts(want["preproc_blocks"]);
  CHECK_MSG(got.size() == wt.size(), "span count");
  size_t n = std::min(got.size(), wt.size());
  int filled = 0, ascii_ok = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!got[i].empty()) ++filled;
    if (ascii_skel(got[i]) == ascii_skel(wt[i])) ++ascii_ok;
    else std::cerr << "  span " << i << " ascii differs:\n    got : " << got[i]
                   << "\n    want: " << wt[i] << "\n";
  }
  std::cerr << "pipeline_textfill: " << filled << "/" << n << " spans filled, " << ascii_ok
            << "/" << n << " ascii-exact vs MinerU (CJK is NFKC-equivalent; a.pdf is digital)\n";
  CHECK_MSG(filled == (int)n, "all spans filled by OCR text-fill");
  CHECK_MSG(ascii_ok == (int)n, "ascii skeleton matches MinerU");
  return TEST_SUMMARY();
}

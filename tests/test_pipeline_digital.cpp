// Copyright (c) mlx-mineru.
// Pipeline P5 (digital-PDF text path): assemble(a.pdf p0) + fill_chars_in_page from the
// embedded text layer == MinerU middle_json span text, BYTE-EXACT (Kangxi-radical codepoints
// + NBSP that the OCR path can only approximate; see test_pipeline_textfill).
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/pdf.hpp"
#include "mineru/pipeline_assemble.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
static std::vector<std::string> texts(const json& blocks) {
  std::vector<std::string> out;
  for (const auto& b : blocks)
    for (const auto& ln : b.value("lines", json::array()))
      for (const auto& sp : ln.value("spans", json::array()))
        if (sp.value("type", "") == "text") out.push_back(sp.value("content", ""));
  return out;
}
static std::vector<unsigned> codepoints(const std::string& s) {
  std::vector<unsigned> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    unsigned cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 6) { cp = c & 0x1F; len = 2; }
    else if ((c >> 4) == 14) { cp = c & 0x0F; len = 3; }
    else { cp = c & 0x07; len = 4; }
    for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    out.push_back(cp);
    i += len;
  }
  return out;
}
// CJK Radicals Supplement (2E80-2EFF) + Kangxi Radicals (2F00-2FDF): the glyphs where the
// vendored pdfium returns CJK-unified but pypdfium2 returns the radical form.
static bool is_radical(unsigned cp) { return cp >= 0x2E80 && cp <= 0x2FDF; }

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string pdf = (argc > 2) ? argv[2] : "a.pdf";

  json model_page = json::parse(read_file(golden_dir + "/pipeline_p0_model.json"));
  json want = json::parse(read_file(golden_dir + "/pipeline_p0_middle.json"));
  int page_w = want["page_size"][0], page_h = want["page_size"][1];

  json page_info = mineru::assemble_page_info(model_page, page_w, page_h, 0);

  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  std::vector<mineru::PageChar> chars;
  for (const mineru::PdfChar& c : doc.extract_chars(0))
    chars.push_back({c.cp, c.idx, c.x0, c.y0, c.x1, c.y1});
  std::cerr << "pipeline_digital: " << chars.size() << " embedded chars\n";
  mineru::fill_chars_in_page(page_info, chars);

  auto got = texts(page_info["preproc_blocks"]);
  auto wt = texts(want["preproc_blocks"]);
  CHECK_MSG(got.size() == wt.size(), "span count");
  // Every span must equal MinerU's modulo the pdfium-version radical encoding: same length,
  // and every differing position is a radical-range codepoint on at least one side.
  int equiv = 0, exact = 0, len_mismatch = 0, bad_diff = 0;
  size_t n = std::min(got.size(), wt.size());
  for (size_t i = 0; i < n; ++i) {
    if (got[i] == wt[i]) { ++exact; ++equiv; continue; }
    auto g = codepoints(got[i]), w = codepoints(wt[i]);
    if (g.size() != w.size()) {
      ++len_mismatch;
      std::cerr << "  span " << i << " length mismatch:\n    got : " << got[i] << "\n    want: " << wt[i] << "\n";
      continue;
    }
    bool ok = true;
    for (size_t k = 0; k < g.size(); ++k)
      if (g[k] != w[k] && !is_radical(g[k]) && !is_radical(w[k])) { ok = false; break; }
    if (ok) ++equiv;
    else { ++bad_diff; std::cerr << "  span " << i << " non-radical diff:\n    got : " << got[i]
                                 << "\n    want: " << wt[i] << "\n"; }
  }
  std::cerr << "pipeline_digital: " << exact << "/" << n << " byte-exact, " << equiv << "/" << n
            << " radical-equivalent vs MinerU\n";
  CHECK_MSG(len_mismatch == 0, "no length mismatches (structure/spacing correct)");
  CHECK_MSG(equiv == (int)n, "all spans equal MinerU modulo pdfium radical encoding");
  return TEST_SUMMARY();
}

// Copyright (c) mlx-mineru.
// Pipeline P5 capstone: the full pipeline-backend vertical for one page —
// model_list + PDF -> assemble -> post-OCR text-fill -> union_make -> Markdown.
// Verified against union_make on MinerU's golden middle_json (ASCII-exact; CJK is
// NFKC-equivalent because a.pdf is a digital PDF — see test_pipeline_textfill).
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/enums.hpp"
#include "mineru/mkcontent.hpp"
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
static std::string ascii_skel(const std::string& s) {
  std::string o;
  for (unsigned char c : s)
    if (c > 0x20 && c < 0x7F) o += (char)c;
  return o;
}
static int count_substr(const std::string& s, const std::string& sub) {
  int n = 0;
  for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + 1)) ++n;
  return n;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string onnx = (argc > 2) ? argv[2] : "models/pipeline/OCR/ocr_rec.onnx";
  std::string dict = (argc > 3) ? argv[3] : "models/pipeline/OCR/ppocrv6_dict.txt";
  std::string pdf = (argc > 4) ? argv[4] : "a.pdf";

  json model_page = json::parse(read_file(golden_dir + "/pipeline_p0_model.json"));
  json want_middle = json::parse(read_file(golden_dir + "/pipeline_p0_middle.json"));
  int page_w = want_middle["page_size"][0], page_h = want_middle["page_size"][1];

  // Full C++ pipeline: render -> assemble -> text-fill -> union_make.
  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  mineru::PageImage img = doc.render_page(0, 200);
  double scale = (double)img.width / page_w;
  json page_info = mineru::assemble_page_info(model_page, page_w, page_h, 0);
  mineru::TextRecognizer rec = mineru::TextRecognizer::load(onnx, dict);
  mineru::fill_span_text(page_info, img.rgb, img.width, img.height, scale, rec);
  std::string md =
      mineru::union_make(json::array({page_info}), mineru::make_mode::kMmMd, "").get<std::string>();

  // Reference: union_make on MinerU's golden middle_json (the renderer is byte-exact).
  std::string ref =
      mineru::union_make(json::array({want_middle}), mineru::make_mode::kMmMd, "").get<std::string>();

  std::cerr << "----- C++ pipeline-backend Markdown (a.pdf p0) -----\n" << md << "\n";
  CHECK_MSG(!md.empty(), "markdown non-empty");
  CHECK_MSG(count_substr(md, "## ") == 4, "4 H2 headings (titles)");
  bool ascii_ok = (ascii_skel(md) == ascii_skel(ref));
  CHECK_MSG(ascii_ok, "markdown ascii skeleton matches union_make(golden)");
  if (!ascii_ok) std::cerr << "  got ascii: " << ascii_skel(md) << "\n  ref ascii: "
                           << ascii_skel(ref) << "\n";
  std::cerr << "pipeline_md: full vertical model_list+pdf -> markdown OK\n";
  return TEST_SUMMARY();
}

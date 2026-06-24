// Copyright (c) mlx-mineru.
// Pipeline P5: multi-page driver — full model_list + PDF -> pdf_info -> Markdown for all
// pages of a.pdf, ASCII-exact vs union_make on MinerU's golden middle_json.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/enums.hpp"
#include "mineru/mkcontent.hpp"
#include "mineru/ocr_rec.hpp"
#include "mineru/pdf.hpp"
#include "mineru/pipeline_driver.hpp"
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

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string onnx = (argc > 2) ? argv[2] : "models/pipeline/OCR/ocr_rec.onnx";
  std::string dict = (argc > 3) ? argv[3] : "models/pipeline/OCR/ppocrv6_dict.txt";
  std::string pdf = (argc > 4) ? argv[4] : "a.pdf";

  json model_list = json::parse(read_file(golden_dir + "/pipeline_model.json"));
  json want = json::parse(read_file(golden_dir + "/pipeline_middle.json"));

  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  std::vector<mineru::PipelinePageImage> pages;
  for (size_t i = 0; i < model_list.size(); ++i) {
    mineru::PageImage im = doc.render_page((int)i, 200);
    mineru::PipelinePageImage pg;
    pg.rgb = std::move(im.rgb);
    pg.w = im.width;
    pg.h = im.height;
    pg.page_w = (int)std::lround(im.width_pt);
    pg.page_h = (int)std::lround(im.height_pt);
    pages.push_back(std::move(pg));
  }

  mineru::TextRecognizer rec = mineru::TextRecognizer::load(onnx, dict);
  json pdf_info = mineru::pipeline_assemble_pages(model_list, pages, rec);

  std::string md = mineru::union_make(pdf_info, mineru::make_mode::kMmMd, "").get<std::string>();
  std::string ref =
      mineru::union_make(want["pdf_info"], mineru::make_mode::kMmMd, "").get<std::string>();

  std::cerr << "===== full a.pdf -> Markdown (" << pdf_info.size() << " pages) =====\n"
            << md << "\n";
  CHECK_MSG((int)pdf_info.size() == (int)model_list.size(), "all pages assembled");
  bool ascii_ok = (ascii_skel(md) == ascii_skel(ref));
  CHECK_MSG(ascii_ok, "full-doc markdown ascii matches union_make(golden)");
  std::cerr << "pipeline_multipage: " << pdf_info.size()
            << " pages model_list+pdf -> Markdown, ascii-exact vs MinerU\n";
  return TEST_SUMMARY();
}

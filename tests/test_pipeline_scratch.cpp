// Copyright (c) mlx-mineru.
// Pipeline P5: fully from-scratch — no MinerU model_list. Render a.pdf p0, build the
// model_list in C++ (layout reading-order + OCR det), assemble + text-fill, render
// Markdown. Verified against union_make on MinerU's golden middle_json (ASCII skeleton).
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/enums.hpp"
#include "mineru/layout_det.hpp"
#include "mineru/mkcontent.hpp"
#include "mineru/ocr_det.hpp"
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
static int count_substr(const std::string& s, const std::string& sub) {
  int n = 0;
  for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + 1)) ++n;
  return n;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string models = (argc > 2) ? argv[2] : "models/pipeline";
  std::string pdf = (argc > 3) ? argv[3] : "a.pdf";

  json want = json::parse(read_file(golden_dir + "/pipeline_p0_middle.json"));
  int page_w = want["page_size"][0], page_h = want["page_size"][1];

  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  mineru::PageImage im = doc.render_page(0, 200);

  // Build the model_list from scratch (no MinerU).
  mineru::LayoutDetector layout = mineru::LayoutDetector::load(models + "/Layout");
  mineru::TextDetector det = mineru::TextDetector::load(models + "/OCR/ocr_det.onnx");
  json page_model = mineru::build_page_model(layout, det, im.rgb, im.width, im.height);
  json model_list = json::array({page_model});
  std::cerr << "scratch: " << page_model["layout_dets"].size() << " layout_dets ("
            << im.width << "x" << im.height << ")\n";

  std::vector<mineru::PipelinePageImage> pages(1);
  pages[0].rgb = im.rgb;
  pages[0].w = im.width;
  pages[0].h = im.height;
  pages[0].page_w = page_w;
  pages[0].page_h = page_h;
  mineru::TextRecognizer rec = mineru::TextRecognizer::load(models + "/OCR/ocr_rec.onnx",
                                                           models + "/OCR/ppocrv6_dict.txt");
  json pdf_info = mineru::pipeline_assemble_pages(model_list, pages, rec);

  std::string md = mineru::union_make(pdf_info, mineru::make_mode::kMmMd, "").get<std::string>();
  std::string ref =
      mineru::union_make(json::array({want}), mineru::make_mode::kMmMd, "").get<std::string>();
  std::cerr << "----- from-scratch Markdown (a.pdf p0) -----\n" << md << "\n";

  CHECK_MSG(!md.empty(), "markdown non-empty");
  CHECK_MSG(count_substr(md, "## ") == 4, "4 H2 headings");
  bool ascii_ok = (ascii_skel(md) == ascii_skel(ref));
  CHECK_MSG(ascii_ok, "from-scratch markdown ascii matches MinerU");
  if (!ascii_ok) std::cerr << "  got: " << ascii_skel(md) << "\n  ref: " << ascii_skel(ref) << "\n";
  std::cerr << "pipeline_scratch: full native PDF->Markdown (no MinerU model_list) OK\n";
  return TEST_SUMMARY();
}

// Copyright (c) mlx-mineru.
// Pipeline P5 (visual spans): from-scratch table recognition on demo1 p5. LayoutDetector +
// per-crop OCR + SLANet+ via build_page_model produce <table>..</table> html. Structure can
// differ from MinerU on the from-scratch crop (OCR/resize variance feeds the cell matcher),
// so we require both tables recognized + at least one exact vs the golden model_list html.
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/layout_det.hpp"
#include "mineru/ocr.hpp"
#include "mineru/ocr_det.hpp"
#include "mineru/pdf.hpp"
#include "mineru/pipeline_driver.hpp"
#include "mineru/table_rec.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
static std::string strip_wrap(std::string h) {
  size_t s = h.find("<table>"), e = h.rfind("</table>");
  return (s != std::string::npos && e != std::string::npos) ? h.substr(s, e + 8 - s) : h;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string models = (argc > 2) ? argv[2] : "models/pipeline";
  std::string pdf = (argc > 3) ? argv[3] : "tests/demo1.pdf";

  json gm = json::parse(read_file(golden_dir + "/demo1_p5_model.json"));
  std::vector<std::string> want;
  for (auto& d : gm["layout_dets"])
    if (d.value("label", "") == "table") want.push_back(strip_wrap(d.value("html", "")));

  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  mineru::PageImage im = doc.render_page(5, 200);
  mineru::LayoutDetector layout = mineru::LayoutDetector::load(models + "/Layout");
  mineru::TextDetector det = mineru::TextDetector::load(models + "/OCR/ocr_det.onnx");
  mineru::OcrPipeline ocr = mineru::OcrPipeline::load(
      models + "/OCR/ocr_det.onnx", models + "/OCR/ocr_rec.onnx", models + "/OCR/ppocrv6_dict.txt");
  mineru::TableRecognizer tr = mineru::TableRecognizer::load(
      models + "/TabRec/SlanetPlus/slanet-plus.onnx",
      models + "/TabRec/SlanetPlus/table_structure_dict.txt");
  json pm = mineru::build_page_model(layout, det, im.rgb, im.width, im.height, nullptr, &ocr, &tr);

  std::vector<std::string> got;
  for (auto& d : pm["layout_dets"])
    if (d.value("label", "") == "table") got.push_back(d.value("html", ""));

  std::cerr << "table_scratch: " << got.size() << " tables recognized (golden " << want.size()
            << ")\n";
  CHECK_MSG(got.size() == want.size(), "table count matches layout");
  int valid = 0, exact = 0;
  for (auto& h : got)
    if (h.rfind("<table>", 0) == 0 && h.find("</table>") != std::string::npos) ++valid;
  for (size_t i = 0; i < got.size(); ++i)
    for (auto& w : want)
      if (got[i] == w) { ++exact; break; }
  std::cerr << "table_scratch: " << valid << " valid <table> html, " << exact
            << " exact vs MinerU\n";
  CHECK_MSG(valid == (int)got.size(), "every table -> valid <table> html");
  CHECK_MSG(exact >= 1, "at least one table html exact vs MinerU");
  return TEST_SUMMARY();
}

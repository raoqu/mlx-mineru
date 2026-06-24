// Copyright (c) mlx-mineru.
// Pipeline P5: C++ LayoutDetector reading-order (PP-DocLayoutV2 order head) reproduces
// MinerU's model_list region dets (label + reading order + bbox) on a.pdf p0.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/layout_det.hpp"
#include "mineru/pdf.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  std::string model_dir = (argc > 1) ? argv[1] : "models/pipeline/Layout";
  std::string golden_dir = (argc > 2) ? argv[2] : "tests/golden";
  std::string pdf = (argc > 3) ? argv[3] : "a.pdf";

  json ml = json::parse(read_file(golden_dir + "/pipeline_p0_model.json"));
  std::vector<json> want;  // region dets (non-ocr_text), in reading order
  for (auto& d : ml["layout_dets"])
    if (d.value("label", "") != "ocr_text") want.push_back(d);
  std::sort(want.begin(), want.end(),
            [](const json& a, const json& b) { return a["index"] < b["index"]; });
  int model_w = ml["page_info"]["width"];

  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf);
  mineru::PageImage img = doc.render_page(0, 200);
  auto got = mineru::LayoutDetector::load(model_dir).detect(img.rgb, img.width, img.height);
  // Keep only region (non-ocr) labels; already in reading order.
  std::vector<mineru::LayoutBox> regions;
  for (auto& b : got) regions.push_back(b);

  std::cerr << "layout_order: model image " << img.width << "x" << img.height << " (golden "
            << model_w << "), " << regions.size() << " dets vs golden " << want.size() << "\n";
  CHECK_MSG(regions.size() == want.size(), "region det count");

  int label_ok = 0, box_ok = 0;
  size_t n = std::min(regions.size(), want.size());
  for (size_t i = 0; i < n; ++i) {
    bool lok = regions[i].label == want[i]["label"].get<std::string>();
    label_ok += lok;
    int dx = 0;
    for (int k = 0; k < 4; ++k)
      dx = std::max(dx, std::abs(regions[i].bbox[k] - want[i]["bbox"][k].get<int>()));
    if (dx <= 3) ++box_ok;
    if (!lok || dx > 3)
      std::cerr << "  det " << i << " got " << regions[i].label << " " << regions[i].bbox[0]
                << "," << regions[i].bbox[1] << " | want " << want[i]["label"] << " maxd=" << dx
                << "\n";
  }
  CHECK_MSG(label_ok == (int)n, "labels + reading order match");
  CHECK_MSG(box_ok == (int)n, "bboxes within 3px (bicubic-resize variance)");
  std::cerr << "layout_order: " << label_ok << "/" << n << " labels in reading order, " << box_ok
            << "/" << n << " boxes within 3px of MinerU\n";
  return TEST_SUMMARY();
}

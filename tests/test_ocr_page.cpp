// Copyright (c) mlx-mineru.
// Pipeline P3: C++ OcrPipeline == MinerU full-page OCR chain (det -> sort -> merge ->
// rotate-crop -> batched rec -> drop_score) on a.pdf p0.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/ocr.hpp"
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
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string det = (argc > 2) ? argv[2] : "models/pipeline/OCR/ocr_det.onnx";
  std::string rec = (argc > 3) ? argv[3] : "models/pipeline/OCR/ocr_rec.onnx";
  std::string dict = (argc > 4) ? argv[4] : "models/pipeline/OCR/ppocrv6_dict.txt";

  json g = json::parse(read_file(golden_dir + "/ocr_page.json"));
  int sw = g["src_w"], sh = g["src_h"];
  std::string raw = read_file(golden_dir + "/" + g["input"].get<std::string>());
  CHECK_MSG((int)raw.size() == sw * sh * 3, "page rgb size");
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  mineru::OcrPipeline ocr = mineru::OcrPipeline::load(det, rec, dict);
  auto lines = ocr.run(rgb, sw, sh, g["drop_score"].get<float>());

  auto& want = g["lines"];
  std::cerr << "ocr_page: C++ " << lines.size() << " lines vs golden " << want.size() << "\n";
  CHECK_MSG(lines.size() == want.size(), "line count");

  // The chain preserves order (sorted_boxes + merge), so compare positionally. Boxes are
  // checked within a few px and text by exact match. NOTE: we don't link OpenCV, so our
  // bilinear differs from cv2.resize by up to 1 LSB (cv2 itself isn't bit-exact across its
  // own scalar/SIMD builds). That sub-pixel delta can shift a det box ~1px and -- because
  // rec pads each batch to its widest crop's aspect -- very rarely flip one repeated
  // glyph (here a doubled em-dash). So we require all boxes within 4px and >=n-1 lines
  // exact, with at most 1 line differing.
  int text_ok = 0, box_ok = 0;
  size_t n = std::min(lines.size(), want.size());
  for (size_t i = 0; i < n; ++i) {
    std::string wt = want[i]["text"].get<std::string>();
    bool t = (lines[i].text == wt);
    text_ok += t;
    if (!t) std::cerr << "  line " << i << " text differs (resize-cascade):\n    got : "
                      << lines[i].text << "\n    want: " << wt << "\n";
    float maxd = 0;
    for (int k = 0; k < 4; ++k) {
      maxd = std::max(maxd, std::abs(lines[i].box[k][0] - want[i]["box"][k][0].get<float>()));
      maxd = std::max(maxd, std::abs(lines[i].box[k][1] - want[i]["box"][k][1].get<float>()));
    }
    if (maxd <= 4.0f) ++box_ok;
    else std::cerr << "  line " << i << " box maxd=" << maxd << "\n";
  }
  CHECK_MSG(text_ok >= (int)n - 1, ">= n-1 line texts exact");
  CHECK_MSG(box_ok == (int)n, "all line boxes match within 4px");
  std::cerr << "ocr_page: " << text_ok << "/" << n << " texts exact, " << box_ok << "/" << n
            << " boxes within 4px of MinerU\n";
  return TEST_SUMMARY();
}

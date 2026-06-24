// Copyright (c) mlx-mineru.
// Pipeline P1: C++ PP-DocLayoutV2 detector == Python core post-process golden
// (scripts/gen_pp_layout_golden.py). Both consume the same saved 800x800 RGB, so
// this verifies onnx inference + box decode/scale/topk/conf/clip exactly.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/layout_det.hpp"
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

  json g = json::parse(read_file(golden_dir + "/layout_det.json"));
  int size = g["size"];
  std::string raw = read_file(golden_dir + "/" + g["input_rgb"].get<std::string>());
  CHECK_MSG((int)raw.size() == size * size * 3, "input rgb size");
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  mineru::LayoutDetector det = mineru::LayoutDetector::load(model_dir, g["conf"].get<float>());
  auto got = det.detect_800(rgb, size, size);

  // detect_800 now returns boxes in reading order; the core-post-process golden is in score
  // order. Compare as a set: sort both by bbox top-left.
  std::sort(got.begin(), got.end(), [](const mineru::LayoutBox& a, const mineru::LayoutBox& b) {
    return a.bbox[1] != b.bbox[1] ? a.bbox[1] < b.bbox[1] : a.bbox[0] < b.bbox[0];
  });
  std::vector<json> want(g["detections"].begin(), g["detections"].end());
  std::sort(want.begin(), want.end(), [](const json& a, const json& b) {
    return a["bbox"][1] != b["bbox"][1] ? a["bbox"][1] < b["bbox"][1] : a["bbox"][0] < b["bbox"][0];
  });
  CHECK_MSG(got.size() == want.size(),
            "count " + std::to_string(got.size()) + " != " + std::to_string(want.size()));
  size_t n = std::min(got.size(), want.size());
  for (size_t i = 0; i < n; ++i) {
    const auto& w = want[i];
    bool label_ok = got[i].label == w["label"].get<std::string>();
    bool box_ok = got[i].bbox[0] == w["bbox"][0].get<int>() && got[i].bbox[1] == w["bbox"][1].get<int>() &&
                  got[i].bbox[2] == w["bbox"][2].get<int>() && got[i].bbox[3] == w["bbox"][3].get<int>();
    bool score_ok = std::abs(got[i].score - w["score"].get<float>()) < 1e-3;
    bool ok = label_ok && box_ok && score_ok;
    CHECK_MSG(ok, "det " + std::to_string(i) + " mismatch");
    if (!ok)
      std::cerr << "  got " << got[i].label << " " << got[i].score << " [" << got[i].bbox[0] << ","
                << got[i].bbox[1] << "," << got[i].bbox[2] << "," << got[i].bbox[3] << "]\n  want "
                << w["label"] << " " << w["score"] << " " << w["bbox"] << "\n";
  }
  std::cerr << "layout_det: " << got.size() << " detections verified vs golden\n";
  return TEST_SUMMARY();
}

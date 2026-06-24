// Copyright (c) mlx-mineru.
// Pipeline P2 (wired tables) — Stage 1: WiredTableRecognizer::segment == MinerU TSRUnet
// (preprocess + unet.onnx -> line segmentation). Compares the 0/1/2 seg map against the
// golden, allowing a small fraction of boundary pixels to differ (bicubic-resize LSB).
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/wired_table.hpp"
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
  std::string onnx = (argc > 2) ? argv[2] : "models/pipeline/TabRec/UnetStructure/unet.onnx";

  json g = json::parse(read_file(golden_dir + "/wired_table.json"));
  int w = g["w"], h = g["h"];
  std::string raw = read_file(golden_dir + "/" + g["input"].get<std::string>());
  CHECK_MSG((int)raw.size() == w * h * 3, "input rgb size");
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  std::string segraw = read_file(golden_dir + "/wired_table_seg.u8");
  std::vector<uint8_t> want(segraw.begin(), segraw.end());

  mineru::WiredTableRecognizer wt = mineru::WiredTableRecognizer::load(onnx);
  int nh, nw;
  auto got = wt.segment(rgb, w, h, nh, nw);
  std::cerr << "wired seg: " << nh << "x" << nw << " (golden " << g["seg_shape"][0] << "x"
            << g["seg_shape"][1] << ")\n";
  CHECK_MSG((int)got.size() == (int)want.size(), "seg size");

  size_t n = std::min(got.size(), want.size()), diff = 0, line_diff = 0;
  for (size_t i = 0; i < n; ++i)
    if (got[i] != want[i]) {
      ++diff;
      if (want[i] != 0 || got[i] != 0) ++line_diff;  // a line pixel disagreement
    }
  double frac = (double)diff / n;
  std::cerr << "wired seg: " << diff << "/" << n << " px differ (" << frac * 100 << "%), "
            << line_diff << " involve line pixels\n";
  // The seg is argmax; near-boundary line pixels can flip with the bicubic-resize LSB.
  CHECK_MSG(frac < 0.01, "segmentation matches MinerU within 1% (boundary LSB)");
  return TEST_SUMMARY();
}

// Copyright (c) mlx-mineru.
// Pipeline P2 (wired tables) — Stage 1: WiredTableRecognizer::segment == MinerU TSRUnet
// (preprocess + unet.onnx -> line segmentation). Compares the 0/1/2 seg map against the
// golden, allowing a small fraction of boundary pixels to differ (bicubic-resize LSB).
#include <algorithm>
#include <array>
#include <cmath>
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

  // Stage 2: cell polygons. Compare against the golden cell bboxes (need_ocr=False ->
  // [x0,y0,x1,y1] per cell). MinerU also emits a few thin edge-sliver cells (between the
  // last gridline and the crop edge) from its line-extension geometry; we verify the main
  // grid cells match within a few px and report the sliver delta.
  std::string praw = read_file(golden_dir + "/wired_table_polys.f32");
  int ncell = (int)praw.size() / (4 * sizeof(float));
  std::vector<std::array<float, 4>> wantc(ncell);
  std::memcpy(wantc.data(), praw.data(), praw.size());

  auto cells = wt.cell_polygons(rgb, w, h);
  std::cerr << "wired cells: " << cells.size() << " (golden " << ncell << ")\n";
  // Each produced cell -> its axis-aligned bbox; match to the nearest golden cell.
  int matched = 0;
  for (const std::array<float, 8>& c : cells) {
    float gx0 = c[0], gy0 = c[1], gx1 = c[0], gy1 = c[1];
    for (int k = 0; k < 4; ++k) {
      gx0 = std::min(gx0, c[2 * k]); gy0 = std::min(gy0, c[2 * k + 1]);
      gx1 = std::max(gx1, c[2 * k]); gy1 = std::max(gy1, c[2 * k + 1]);
    }
    float best = 1e9;
    for (auto& wc : wantc)
      best = std::min(best, std::abs(gx0 - wc[0]) + std::abs(gy0 - wc[1]) + std::abs(gx1 - wc[2]) +
                                std::abs(gy1 - wc[3]));
    if (best <= 12) ++matched;  // ~3px/edge
  }
  std::cerr << "wired cells: " << matched << "/" << cells.size()
            << " match a golden cell within ~3px/edge\n";
  CHECK_MSG(cells.size() >= 12, "extracts a plausible cell grid");
  CHECK_MSG(matched >= (int)cells.size() - 1, "produced cells align with MinerU's grid");
  return TEST_SUMMARY();
}

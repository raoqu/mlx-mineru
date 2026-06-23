// Copyright (c) mlx-mineru.
// Phase 4c verification: Qwen2-VL image preprocessing matches transformers
// Qwen2VLImageProcessor (slow / PIL-bicubic path).
//   - smart_resize: exact integer parity on several aspect ratios.
//   - full pipeline: grid_thw exact; pixel_values mean/std/min/max + 256 sampled
//     values within tolerance (PIL fixed-point bicubic ported faithfully).
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/image_preprocess.hpp"
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
  json g = json::parse(read_file(golden_dir + "/preprocess.json"));

  int factor = g["config"]["patch_size"].get<int>() * g["config"]["merge_size"].get<int>();
  int min_px = g["config"]["min_pixels"], max_px = g["config"]["max_pixels"];

  // smart_resize parity.
  for (auto& c : g["smart_resize"]) {
    if (c.contains("error")) continue;
    auto hw = mineru::smart_resize(c["h"], c["w"], factor, min_px, max_px);
    CHECK_MSG(hw[0] == c["h_bar"].get<int>() && hw[1] == c["w_bar"].get<int>(),
              "smart_resize(" + std::to_string(c["h"].get<int>()) + "," +
                  std::to_string(c["w"].get<int>()) + ") got " + std::to_string(hw[0]) + "x" +
                  std::to_string(hw[1]));
  }

  // Full pipeline on identical raw RGB input.
  std::string raw = read_file(golden_dir + "/" + g["input_rgb"].get<std::string>());
  int w = g["in_w"], h = g["in_h"];
  std::vector<uint8_t> rgb(raw.begin(), raw.end());
  CHECK_MSG((int)rgb.size() == w * h * 3, "input rgb size");

  mineru::VisionInput vi = mineru::preprocess_image(rgb, w, h);

  CHECK_MSG(vi.grid_thw[0] == g["grid_thw"][0].get<int>() &&
                vi.grid_thw[1] == g["grid_thw"][1].get<int>() &&
                vi.grid_thw[2] == g["grid_thw"][2].get<int>(),
            "grid_thw mismatch");
  CHECK(vi.seq_len() == g["seq_len"].get<int>());
  CHECK(vi.feat_dim() == g["feat_dim"].get<int>());

  // Global stats.
  const auto& pv = vi.pixel_values;
  double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
  for (float v : pv) { sum += v; sq += (double)v * v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
  double mean = sum / pv.size();
  double stdev = std::sqrt(sq / pv.size() - mean * mean);
  CHECK_MSG(std::abs(mean - g["mean"].get<double>()) < 1e-3, "mean " + std::to_string(mean));
  CHECK_MSG(std::abs(stdev - g["std"].get<double>()) < 1e-3, "std " + std::to_string(stdev));
  CHECK_MSG(std::abs(mn - g["min"].get<double>()) < 5e-3, "min " + std::to_string(mn));
  CHECK_MSG(std::abs(mx - g["max"].get<double>()) < 5e-3, "max " + std::to_string(mx));

  // Sampled values.
  double max_diff = 0;
  int bad = 0;
  for (auto& s : g["samples"]) {
    size_t idx = s[0].get<size_t>();
    double want = s[1].get<double>();
    if (idx >= pv.size()) { ++bad; continue; }
    double diff = std::abs(pv[idx] - want);
    max_diff = std::max(max_diff, diff);
    if (diff > 3e-3) ++bad;
  }
  CHECK_MSG(bad == 0, std::to_string(bad) + " samples exceed tol; max_diff=" + std::to_string(max_diff));
  std::cerr << "preprocess: max sample diff = " << max_diff << "\n";

  return TEST_SUMMARY();
}

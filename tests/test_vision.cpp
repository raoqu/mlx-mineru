// Copyright (c) mlx-mineru.
// Phase 4d (vision) verification: the MLX C++ Qwen2-VL vision tower matches the
// transformers vision encoder. Feeds the bit-exact preprocessed deterministic
// image; compares merged image-embeds shape exactly and values within tolerance
// (bf16 op-ordering differences).
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/image_preprocess.hpp"
#include "mineru/qwen2_vl.hpp"
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
  std::string weights = (argc > 1) ? argv[1] : "models/MinerU2.5-tokenizer/model.safetensors";
  std::string golden_dir = (argc > 2) ? argv[2] : "tests/golden";

  json pg = json::parse(read_file(golden_dir + "/preprocess.json"));
  std::string raw = read_file(golden_dir + "/" + pg["input_rgb"].get<std::string>());
  int w = pg["in_w"], h = pg["in_h"];
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  mineru::VisionInput vi = mineru::preprocess_image(rgb, w, h);

  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(weights);
  std::vector<float> embeds = model.forward_vision(vi.pixel_values, vi.grid_thw);

  json g = json::parse(read_file(golden_dir + "/vision.json"));
  int rows = g["embed_shape"][0], cols = g["embed_shape"][1];
  CHECK_MSG((int)embeds.size() == rows * cols, "embed size " + std::to_string(embeds.size()) +
                                                   " != " + std::to_string(rows * cols));

  double sum = 0, sq = 0, mn = 1e30, mx = -1e30;
  for (float v : embeds) { sum += v; sq += (double)v * v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
  double mean = sum / embeds.size();
  double stdev = std::sqrt(sq / embeds.size() - mean * mean);
  double dmean = std::abs(mean - g["mean"].get<double>());
  double dstd = std::abs(stdev - g["std"].get<double>());
  CHECK_MSG(dmean < 0.5, "mean diff " + std::to_string(dmean) + " (got " + std::to_string(mean) + ")");
  CHECK_MSG(dstd / std::abs(g["std"].get<double>()) < 0.03, "std rel diff (got " + std::to_string(stdev) + ")");

  // Per-sample comparison via mean-absolute-error relative to std (robust to a
  // few bf16-accumulation outliers after 32 transformer layers; a real bug shifts
  // the whole distribution, which the mean/std checks above would catch). Also
  // require the bulk of samples to be within a tight relative bound.
  double max_abs = 0, sum_abs = 0;
  int n = 0, within = 0;
  for (auto& s : g["samples"]) {
    size_t idx = s[0].get<size_t>();
    double want = s[1].get<double>();
    if (idx >= embeds.size()) continue;
    double diff = std::abs(embeds[idx] - want);
    max_abs = std::max(max_abs, diff);
    sum_abs += diff;
    ++n;
    if (diff < 3.0 || diff < 0.1 * std::abs(want)) ++within;
  }
  double mae_over_std = (sum_abs / n) / stdev;
  CHECK_MSG(mae_over_std < 0.05, "vision MAE/std " + std::to_string(mae_over_std));
  CHECK_MSG(within >= 0.9 * n, "only " + std::to_string(within) + "/" + std::to_string(n) +
                                   " samples within tol");
  std::cerr << "vision: mean " << mean << " std " << stdev << " MAE/std " << mae_over_std
            << " within " << within << "/" << n << " max_abs " << max_abs << "\n";

  return TEST_SUMMARY();
}

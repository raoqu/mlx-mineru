// Copyright (c) mlx-mineru.
// Phase 4d (end-to-end) verification: the full MLX C++ multimodal forward
// (preprocess -> vision -> merge -> LLM) matches the transformers model. At each
// greedy step (driven from reference tokens) the argmax must match the reference,
// except at bf16 near-ties (small top-2 gap).
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

  json g = json::parse(read_file(golden_dir + "/vlm.json"));
  std::vector<int> input_ids = g["input_ids"].get<std::vector<int>>();
  int n_img = g["n_img"];
  std::array<int, 3> grid{g["grid_thw"][0], g["grid_thw"][1], g["grid_thw"][2]};

  mineru::VisionInput vi = mineru::preprocess_image(rgb, w, h);
  CHECK_MSG(vi.grid_thw == grid, "grid mismatch");

  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(weights);
  std::vector<float> embeds = model.forward_vision(vi.pixel_values, vi.grid_thw);
  CHECK_MSG((int)embeds.size() == n_img * model.config().hidden_size, "embed size");

  const double kTieGap = 0.25;
  std::vector<int> cur = input_ids;
  int step = 0;
  for (auto& s : g["greedy"]) {
    int want = s["token"];
    double gap = s["gap"];
    std::vector<float> logits = model.multimodal_last_logits(cur, embeds, n_img, grid);
    int got = 0;
    for (int i = 1; i < (int)logits.size(); ++i)
      if (logits[i] > logits[got]) got = i;
    bool ok = (got == want) || (gap < kTieGap);
    CHECK_MSG(ok, "step " + std::to_string(step) + ": got " + std::to_string(got) + " want " +
                      std::to_string(want) + " (gap=" + std::to_string(gap) + ")");
    cur.push_back(want);
    ++step;
  }
  std::cerr << "vlm: verified " << step << " greedy steps end-to-end\n";

  return TEST_SUMMARY();
}

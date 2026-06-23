// Copyright (c) mlx-mineru.
// Verify batched multimodal generation == sequential generation (per sample).
// Uses the deterministic preprocessed image's vision embeds with several
// different-length prompts, so left-padding + masking are exercised.
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

  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(weights);  // bits=0 (deterministic)
  mineru::VisionInput vi = mineru::preprocess_image(rgb, w, h);
  std::vector<float> embeds = model.forward_vision(vi.pixel_values, vi.grid_thw);
  int n_img = vi.seq_len() / (model.config().spatial_merge_size * model.config().spatial_merge_size);
  int IMG = model.config().image_token_id;

  // Several prompts of DIFFERENT lengths, each with the image-token run.
  auto make = [&](std::vector<int> pre, std::vector<int> post) {
    std::vector<int> ids = pre;
    ids.insert(ids.end(), n_img, IMG);
    ids.insert(ids.end(), post.begin(), post.end());
    return ids;
  };
  std::vector<std::vector<int>> prompts = {
      make({151644, 8948, 198}, {151645, 198, 151644, 77091, 198}),
      make({151644, 872, 198}, {198, 1782, 21300, 25, 151645, 198, 151644, 77091, 198}),
      make({151644, 8948, 198, 2610}, {25, 151645, 198, 151644, 77091, 198}),
  };
  int B = (int)prompts.size();
  std::vector<std::vector<float>> emb_list(B, embeds);
  std::vector<int> nimgs(B, n_img);
  std::vector<std::array<int, 3>> grids(B, vi.grid_thw);
  std::vector<int> eos = {151645, 151643};
  int max_new = 12;

  // Sequential reference.
  std::vector<std::vector<int>> seq(B);
  for (int b = 0; b < B; ++b)
    seq[b] = model.generate_multimodal(prompts[b], embeds, n_img, vi.grid_thw, max_new, eos);

  // Batched.
  std::vector<std::vector<int>> bat =
      model.generate_multimodal_batch(prompts, emb_list, nimgs, grids, max_new, eos);

  CHECK(bat.size() == (size_t)B);
  for (int b = 0; b < B; ++b) {
    bool ok = (bat[b] == seq[b]);
    CHECK_MSG(ok, "sample " + std::to_string(b) + ": batched != sequential");
    if (!ok) {
      std::cerr << "  seq: "; for (int x : seq[b]) std::cerr << x << " ";
      std::cerr << "\n  bat: "; for (int x : bat[b]) std::cerr << x << " ";
      std::cerr << "\n";
    }
  }
  std::cerr << "vlm_batch: verified " << B << " samples (batched == sequential)\n";
  return TEST_SUMMARY();
}

// Copyright (c) mlx-mineru.
// Qwen2-VL model in MLX C++ — faithful port of mlx-vlm's qwen2_vl
// (language.py / vision.py). Phase 4d. Weights loaded from the model's
// safetensors. This header exposes the language-model forward (verifiable with
// text-only input) and will grow the vision tower + multimodal merge.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

struct Qwen2VLConfig {
  // text_config
  int hidden_size = 896;
  int num_hidden_layers = 24;
  int num_attention_heads = 14;
  int num_key_value_heads = 2;
  int intermediate_size = 4864;
  int vocab_size = 151936;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  std::vector<int> mrope_section = {8, 12, 12};
  bool tie_word_embeddings = true;
  // special token ids
  int image_token_id = 151655;
  int vision_start_token_id = 151652;
  int spatial_merge_size = 2;
  int head_dim() const { return hidden_size / num_attention_heads; }
};

// 3D MRoPE position ids, shape [3][seq].
using PositionIds = std::array<std::vector<int>, 3>;

class Qwen2VLModel {
 public:
  static Qwen2VLModel load(const std::string& weights_path, const Qwen2VLConfig& cfg = {});
  ~Qwen2VLModel();
  Qwen2VLModel(Qwen2VLModel&&) noexcept;
  Qwen2VLModel& operator=(Qwen2VLModel&&) noexcept;

  const Qwen2VLConfig& config() const;

  // Text-only forward over `tokens`; returns logits for the LAST position
  // (float32, length vocab_size). position_ids default to 0..L-1 on all 3 axes.
  std::vector<float> forward_text_logits(const std::vector<int>& tokens) const;

  // Greedy-decode the argmax next-token id for `tokens` (text-only).
  int argmax_next(const std::vector<int>& tokens) const;

 private:
  Qwen2VLModel();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

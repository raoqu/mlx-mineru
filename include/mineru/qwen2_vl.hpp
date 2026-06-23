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
  // Weight quantization for the LLM path (the per-token generation bottleneck).
  // 0 = full bf16 (used by parity tests); 4 or 8 = 4/8-bit affine quant of the
  // decoder linear weights + tied embeddings (~4x less memory bandwidth -> faster
  // generation on Apple Silicon). Vision tower stays bf16 (runs once per image).
  int quantize_bits = 0;
  int q_group_size = 64;
  // special token ids
  int image_token_id = 151655;
  int vision_start_token_id = 151652;
  int spatial_merge_size = 2;
  int head_dim() const { return hidden_size / num_attention_heads; }
  // vision_config
  int v_embed_dim = 1280;
  int v_depth = 32;
  int v_num_heads = 16;
  int v_patch_size = 14;
  int v_temporal_patch_size = 2;
  int v_mlp_ratio = 4;
  float v_rope_theta = 10000.0f;
  int v_head_dim() const { return v_embed_dim / v_num_heads; }  // 80
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

  // Vision tower: pixel_values [seq*1176] (row-major, seq = t*h*w patches) and
  // grid (t,h,w). Returns merged image embeds, flattened [seq/merge^2, hidden_size].
  std::vector<float> forward_vision(const std::vector<float>& pixel_values,
                                    const std::array<int, 3>& grid) const;

  // Encode several crops at once: builds all vision graphs and evaluates them in
  // a single synchronized batch (MLX overlaps the independent forwards on the
  // GPU), instead of one blocking forward+sync per crop. Returns per-crop embeds.
  std::vector<std::vector<float>> forward_vision_batch(
      const std::vector<std::vector<float>>& pixel_values_list,
      const std::vector<std::array<int, 3>>& grids) const;

  // Multimodal last-token logits (float32, vocab): `input_ids` contains a
  // contiguous run of image_token_id placeholders replaced by `image_embeds`
  // ([n_img, hidden_size]); `grid` is the image (t,gh,gw).
  std::vector<float> multimodal_last_logits(const std::vector<int>& input_ids,
                                            const std::vector<float>& image_embeds, int n_img,
                                            const std::array<int, 3>& grid) const;

  // Greedy multimodal generation; appends argmax tokens until `max_new_tokens`
  // or an eos id. Returns the generated token ids (excluding the prompt).
  std::vector<int> generate_multimodal(const std::vector<int>& input_ids,
                                       const std::vector<float>& image_embeds, int n_img,
                                       const std::array<int, 3>& grid, int max_new_tokens,
                                       const std::vector<int>& eos_ids) const;

  // Batched greedy multimodal generation: runs all samples through one batched
  // forward (left-padded), decoding together with per-sample EOS. Equivalent to
  // calling generate_multimodal on each sample, but far faster (weights read
  // ~max-length times instead of sum-of-lengths). Returns per-sample generated ids.
  std::vector<std::vector<int>> generate_multimodal_batch(
      const std::vector<std::vector<int>>& input_ids,
      const std::vector<std::vector<float>>& image_embeds, const std::vector<int>& n_img,
      const std::vector<std::array<int, 3>>& grids, int max_new_tokens,
      const std::vector<int>& eos_ids) const;

 private:
  Qwen2VLModel();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

// Copyright (c) mlx-mineru.
// MLX/Metal batched greedy decoder for the UniMERNet custom-mBART formula decoder. Replaces the
// per-token ORT decode (the dominant cost in formula recognition) with a single batched
// autoregressive loop on the GPU: all formulas on a page decode together (batch B), self-attention
// KV is cached, and cross-attention K/V is computed ONCE per formula from the encoder output.
// Weights come from mfr_decoder.safetensors (+ mfr_decoder_config.json), produced by
// scripts/extract_mfr_decoder_weights.py. load() returns nullptr if MLX is unavailable or the
// files are missing, so callers fall back to the ORT decoder.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mineru {

class MfrMlxDecoder {
 public:
  // Load weights + config; nullptr on any failure.
  static std::unique_ptr<MfrMlxDecoder> load(const std::string& safetensors_path,
                                             const std::string& config_json_path);
  ~MfrMlxDecoder();

  // Batched greedy decode. enc[b] is formula b's encoder hidden state, flattened row-major as
  // [N[b], d_model]; N[b] is its token count. Returns B id sequences (excluding BOS, truncated at
  // the first EOS per sequence). If force_steps > 0, decodes exactly that many steps ignoring EOS
  // (for validation).
  std::vector<std::vector<int>> decode(const std::vector<std::vector<float>>& enc,
                                       const std::vector<int>& N, int max_new_tokens,
                                       int force_steps = 0) const;

 private:
  MfrMlxDecoder();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Qwen2-VL language model forward in MLX C++ (faithful to mlx-vlm qwen2_vl).
#include "mineru/qwen2_vl.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "mlx/mlx.h"

namespace mineru {
namespace {
namespace mx = mlx::core;
using mx::array;

array linear(const array& x, const array& w, const std::optional<array>& b = std::nullopt) {
  array y = mx::matmul(x, mx::transpose(w));  // w: (out, in)
  if (b) y = mx::add(y, *b);
  return y;
}

// rotate_half (NeoX / half-split): concat(-x2, x1).
array rotate_half(const array& x) {
  auto parts = mx::split(x, 2, /*axis=*/-1);
  return mx::concatenate({mx::negative(parts[1]), parts[0]}, -1);
}

}  // namespace

struct Qwen2VLModel::Impl {
  Qwen2VLConfig cfg;
  std::unordered_map<std::string, array> w;

  const array& get(const std::string& name) const {
    auto it = w.find(name);
    if (it == w.end()) throw std::runtime_error("qwen2_vl: missing weight " + name);
    return it->second;
  }

  // Build (cos, sin) of shape (1,1,L,head_dim) in bf16 from 3D position ids.
  std::pair<array, array> rope_cos_sin(const PositionIds& pos, int L) const {
    int hd = cfg.head_dim();      // 64
    int half = hd / 2;            // 32
    // per-frequency axis selector (chunked mrope_section [8,12,12]).
    std::vector<int> sel(half, 0);
    int off = cfg.mrope_section[0];
    for (int idx = off; idx < off + cfg.mrope_section[1] && idx < half; ++idx) sel[idx] = 1;
    off += cfg.mrope_section[1];
    for (int idx = off; idx < off + cfg.mrope_section[2] && idx < half; ++idx) sel[idx] = 2;

    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i)
      inv_freq[i] = 1.0f / std::pow(cfg.rope_theta, (2.0f * i) / hd);

    // freqs[t, f] = pos[sel[f]][t] * inv_freq[f]
    std::vector<float> emb(static_cast<size_t>(L) * hd);
    for (int t = 0; t < L; ++t) {
      for (int f = 0; f < half; ++f) {
        float angle = static_cast<float>(pos[sel[f]][t]) * inv_freq[f];
        emb[static_cast<size_t>(t) * hd + f] = angle;          // first half
        emb[static_cast<size_t>(t) * hd + half + f] = angle;   // duplicated half
      }
    }
    array embA(emb.data(), mx::Shape{L, hd}, mx::float32);
    array cos = mx::reshape(mx::cos(embA), {1, 1, L, hd});
    array sin = mx::reshape(mx::sin(embA), {1, 1, L, hd});
    return {mx::astype(cos, mx::bfloat16), mx::astype(sin, mx::bfloat16)};
  }

  array apply_rope(const array& x, const array& cos, const array& sin) const {
    return mx::add(mx::multiply(x, cos), mx::multiply(rotate_half(x), sin));
  }

  // Full transformer forward; returns final-norm hidden states (1, L, H).
  array forward_hidden(const std::vector<int>& tokens, const PositionIds& pos) const {
    int L = static_cast<int>(tokens.size());
    int H = cfg.hidden_size, nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads;
    int hd = cfg.head_dim();
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::vector<int> tok = tokens;
    array idx(tok.data(), mx::Shape{L}, mx::int32);
    array h = mx::reshape(mx::take(get("model.embed_tokens.weight"), idx, 0), {1, L, H});

    auto [cosA, sinA] = rope_cos_sin(pos, L);

    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
      std::string p = "model.layers." + std::to_string(l) + ".";
      array hn = mx::fast::rms_norm(h, get(p + "input_layernorm.weight"), cfg.rms_norm_eps);

      array q = linear(hn, get(p + "self_attn.q_proj.weight"), get(p + "self_attn.q_proj.bias"));
      array k = linear(hn, get(p + "self_attn.k_proj.weight"), get(p + "self_attn.k_proj.bias"));
      array v = linear(hn, get(p + "self_attn.v_proj.weight"), get(p + "self_attn.v_proj.bias"));

      q = mx::transpose(mx::reshape(q, {1, L, nH, hd}), {0, 2, 1, 3});
      k = mx::transpose(mx::reshape(k, {1, L, nKV, hd}), {0, 2, 1, 3});
      v = mx::transpose(mx::reshape(v, {1, L, nKV, hd}), {0, 2, 1, 3});

      q = apply_rope(q, cosA, sinA);
      k = apply_rope(k, cosA, sinA);

      array o = mx::fast::scaled_dot_product_attention(q, k, v, scale, "causal");
      o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {1, L, nH * hd});
      o = linear(o, get(p + "self_attn.o_proj.weight"));
      h = mx::add(h, o);

      array h2 = mx::fast::rms_norm(h, get(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);
      array gate = linear(h2, get(p + "mlp.gate_proj.weight"));
      array up = linear(h2, get(p + "mlp.up_proj.weight"));
      array silu = mx::multiply(gate, mx::sigmoid(gate));
      array mlp = linear(mx::multiply(silu, up), get(p + "mlp.down_proj.weight"));
      h = mx::add(h, mlp);
    }
    return mx::fast::rms_norm(h, get("model.norm.weight"), cfg.rms_norm_eps);
  }

  std::vector<float> last_logits(const std::vector<int>& tokens, const PositionIds& pos) const {
    int L = static_cast<int>(tokens.size());
    array h = forward_hidden(tokens, pos);
    array h_last = mx::take(h, L - 1, /*axis=*/1);  // (1, H)
    // tied embeddings: logits = h_last @ embed_tokens.weight.T
    array logits = linear(h_last, get("model.embed_tokens.weight"));  // (1, vocab)
    logits = mx::astype(logits, mx::float32);
    mx::eval(logits);
    const float* p = logits.data<float>();
    return std::vector<float>(p, p + cfg.vocab_size);
  }
};

Qwen2VLModel::Qwen2VLModel() : impl_(std::make_unique<Impl>()) {}
Qwen2VLModel::~Qwen2VLModel() = default;
Qwen2VLModel::Qwen2VLModel(Qwen2VLModel&&) noexcept = default;
Qwen2VLModel& Qwen2VLModel::operator=(Qwen2VLModel&&) noexcept = default;

Qwen2VLModel Qwen2VLModel::load(const std::string& weights_path, const Qwen2VLConfig& cfg) {
  Qwen2VLModel m;
  m.impl_->cfg = cfg;
  auto loaded = mx::load_safetensors(weights_path);
  m.impl_->w = std::move(loaded.first);
  return m;
}

const Qwen2VLConfig& Qwen2VLModel::config() const { return impl_->cfg; }

std::vector<float> Qwen2VLModel::forward_text_logits(const std::vector<int>& tokens) const {
  int L = static_cast<int>(tokens.size());
  PositionIds pos;
  for (int a = 0; a < 3; ++a) {
    pos[a].resize(L);
    for (int t = 0; t < L; ++t) pos[a][t] = t;
  }
  return impl_->last_logits(tokens, pos);
}

int Qwen2VLModel::argmax_next(const std::vector<int>& tokens) const {
  std::vector<float> logits = forward_text_logits(tokens);
  int best = 0;
  for (int i = 1; i < (int)logits.size(); ++i)
    if (logits[i] > logits[best]) best = i;
  return best;
}

}  // namespace mineru

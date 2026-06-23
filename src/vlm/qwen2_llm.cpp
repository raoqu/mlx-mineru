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

  // Embed token ids -> (1, L, H).
  array embed_tokens(const std::vector<int>& tokens) const {
    int L = static_cast<int>(tokens.size());
    std::vector<int> tok = tokens;
    array idx(tok.data(), mx::Shape{L}, mx::int32);
    return mx::reshape(mx::take(get("model.embed_tokens.weight"), idx, 0), {1, L, cfg.hidden_size});
  }

  // 3D MRoPE position ids for a sequence with at most one image (Qwen2-VL
  // get_rope_index). `grid` is (t, gh, gw) in patch units; image tokens must be
  // the contiguous run of image_token_id.
  PositionIds get_rope_index(const std::vector<int>& ids, const std::array<int, 3>& grid) const {
    int L = static_cast<int>(ids.size());
    PositionIds pos;
    for (int a = 0; a < 3; ++a) pos[a].assign(L, 0);
    int img_start = -1;
    for (int i = 0; i < L; ++i)
      if (ids[i] == cfg.image_token_id) { img_start = i; break; }
    if (img_start < 0) {
      for (int a = 0; a < 3; ++a)
        for (int t = 0; t < L; ++t) pos[a][t] = t;
      return pos;
    }
    int merge = cfg.spatial_merge_size;
    int lt = grid[0], lh = grid[1] / merge, lw = grid[2] / merge;
    int n_img = lt * lh * lw;
    for (int t = 0; t < img_start; ++t)
      for (int a = 0; a < 3; ++a) pos[a][t] = t;       // text before image
    int base = img_start;                              // text_len + st_idx(0)
    for (int k = 0; k < n_img; ++k) {
      int ti = k / (lh * lw), hi = (k / lw) % lh, wi = k % lw;
      pos[0][img_start + k] = base + ti;
      pos[1][img_start + k] = base + hi;
      pos[2][img_start + k] = base + wi;
    }
    int next = base + std::max(std::max(lt, lh), lw);  // max image pos + 1
    int after = img_start + n_img;
    for (int j = after; j < L; ++j)
      for (int a = 0; a < 3; ++a) pos[a][j] = next + (j - after);
    return pos;
  }

  // Full transformer forward from input embeddings (1,L,H) + 3D positions.
  array forward_hidden_from_embeds(array h, const PositionIds& pos) const {
    int L = h.shape()[1];
    int nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads;
    int hd = cfg.head_dim();
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
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

  array forward_hidden(const std::vector<int>& tokens, const PositionIds& pos) const {
    return forward_hidden_from_embeds(embed_tokens(tokens), pos);
  }

  // Build inputs_embeds replacing the contiguous image-token run with vision
  // embeds (image_embeds: [n_img, H] row-major float).
  array build_multimodal_embeds(const std::vector<int>& ids,
                                const std::vector<float>& image_embeds, int n_img) const {
    int L = static_cast<int>(ids.size()), H = cfg.hidden_size;
    array text = mx::reshape(embed_tokens(ids), {L, H});
    int img_start = -1;
    for (int i = 0; i < L; ++i)
      if (ids[i] == cfg.image_token_id) { img_start = i; break; }
    if (img_start < 0) return mx::reshape(text, {1, L, H});
    array img = mx::astype(array(image_embeds.data(), mx::Shape{n_img, H}, mx::float32), mx::bfloat16);
    std::vector<array> parts;
    if (img_start > 0) parts.push_back(mx::slice(text, {0, 0}, {img_start, H}));
    parts.push_back(img);
    if (img_start + n_img < L) parts.push_back(mx::slice(text, {img_start + n_img, 0}, {L, H}));
    return mx::reshape(mx::concatenate(parts, 0), {1, L, H});
  }

  // ---- Vision tower (faithful to mlx-vlm qwen2_vl vision.py) ----------------
  array vision_rope_apply(const array& x, const array& cos, const array& sin) const {
    // x: (seq, heads, head_dim); cos/sin: (seq, 1, head_dim) broadcast over heads.
    return mx::add(mx::multiply(x, cos), mx::multiply(rotate_half(x), sin));
  }

  std::vector<float> forward_vision(const std::vector<float>& pixel_values,
                                    const std::array<int, 3>& grid) const {
    int gt = grid[0], gh = grid[1], gw = grid[2];
    int seq = gt * gh * gw;
    int ed = cfg.v_embed_dim, nH = cfg.v_num_heads, hd = cfg.v_head_dim();  // 1280,16,80
    int ms = cfg.spatial_merge_size;
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // patch_embed as matmul: weight [ed,3,2,14,14] -> [ed, 1176].
    int feat = 3 * cfg.v_temporal_patch_size * cfg.v_patch_size * cfg.v_patch_size;  // 1176
    array pv(pixel_values.data(), mx::Shape{seq, feat}, mx::float32);
    array pe_w = mx::reshape(get("visual.patch_embed.proj.weight"), {ed, feat});
    array h = mx::matmul(mx::astype(pv, mx::bfloat16), mx::transpose(pe_w));  // (seq, ed)

    // Vision 2D rotary freqs, in merge-block patch order (matches patchify).
    int rdim = hd / 2;            // 40
    int nfreq = rdim / 2;         // 20
    std::vector<float> inv_freq(nfreq);
    for (int i = 0; i < nfreq; ++i)
      inv_freq[i] = 1.0f / std::pow(cfg.v_rope_theta, (2.0f * i) / rdim);
    std::vector<float> cosv(static_cast<size_t>(seq) * hd), sinv(static_cast<size_t>(seq) * hd);
    int p = 0;
    for (int t = 0; t < gt; ++t)
      for (int hb = 0; hb < gh / ms; ++hb)
        for (int wb = 0; wb < gw / ms; ++wb)
          for (int mh = 0; mh < ms; ++mh)
            for (int mw = 0; mw < ms; ++mw) {
              int hp = hb * ms + mh, wp = wb * ms + mw;
              for (int f = 0; f < nfreq; ++f) {
                float ah = hp * inv_freq[f], aw = wp * inv_freq[f];
                // freqs = [h-freqs(20), w-freqs(20)] then tiled x2 -> 80.
                size_t base = static_cast<size_t>(p) * hd;
                cosv[base + f] = std::cos(ah);             cosv[base + nfreq + f] = std::cos(aw);
                cosv[base + rdim + f] = std::cos(ah);      cosv[base + rdim + nfreq + f] = std::cos(aw);
                sinv[base + f] = std::sin(ah);             sinv[base + nfreq + f] = std::sin(aw);
                sinv[base + rdim + f] = std::sin(ah);      sinv[base + rdim + nfreq + f] = std::sin(aw);
              }
              ++p;
            }
    array cos = mx::astype(mx::reshape(array(cosv.data(), mx::Shape{seq, hd}, mx::float32), {seq, 1, hd}), mx::bfloat16);
    array sin = mx::astype(mx::reshape(array(sinv.data(), mx::Shape{seq, hd}, mx::float32), {seq, 1, hd}), mx::bfloat16);

    for (int b = 0; b < cfg.v_depth; ++b) {
      std::string pr = "visual.blocks." + std::to_string(b) + ".";
      array hn = mx::fast::layer_norm(h, get(pr + "norm1.weight"), get(pr + "norm1.bias"), 1e-6f);
      array qkv = linear(hn, get(pr + "attn.qkv.weight"), get(pr + "attn.qkv.bias"));  // (seq, 3*ed)
      qkv = mx::reshape(qkv, {seq, 3, nH, hd});
      array q = mx::squeeze(mx::slice(qkv, {0, 0, 0, 0}, {seq, 1, nH, hd}), 1);  // (seq,nH,hd)
      array k = mx::squeeze(mx::slice(qkv, {0, 1, 0, 0}, {seq, 2, nH, hd}), 1);
      array v = mx::squeeze(mx::slice(qkv, {0, 2, 0, 0}, {seq, 3, nH, hd}), 1);
      q = vision_rope_apply(q, cos, sin);
      k = vision_rope_apply(k, cos, sin);
      // (seq,nH,hd) -> (1,nH,seq,hd)
      q = mx::expand_dims(mx::transpose(q, {1, 0, 2}), 0);
      k = mx::expand_dims(mx::transpose(k, {1, 0, 2}), 0);
      v = mx::expand_dims(mx::transpose(v, {1, 0, 2}), 0);
      array o = mx::fast::scaled_dot_product_attention(q, k, v, scale, "");  // full attn
      o = mx::reshape(mx::transpose(mx::squeeze(o, 0), {1, 0, 2}), {seq, ed});
      o = linear(o, get(pr + "attn.proj.weight"), get(pr + "attn.proj.bias"));
      h = mx::add(h, o);

      array h2 = mx::fast::layer_norm(h, get(pr + "norm2.weight"), get(pr + "norm2.bias"), 1e-6f);
      array fc1 = linear(h2, get(pr + "mlp.fc1.weight"), get(pr + "mlp.fc1.bias"));
      // quick_gelu: x * sigmoid(1.702 x)
      array act = mx::multiply(fc1, mx::sigmoid(mx::multiply(fc1, array(1.702f))));
      array fc2 = linear(act, get(pr + "mlp.fc2.weight"), get(pr + "mlp.fc2.bias"));
      h = mx::add(h, fc2);
    }

    // PatchMerger: ln_q -> reshape (seq/ms^2, ed*ms^2) -> Linear -> GELU -> Linear.
    array m = mx::fast::layer_norm(h, get("visual.merger.ln_q.weight"), get("visual.merger.ln_q.bias"), 1e-6f);
    int merged = seq / (ms * ms);
    m = mx::reshape(m, {merged, ed * ms * ms});
    m = linear(m, get("visual.merger.mlp.0.weight"), get("visual.merger.mlp.0.bias"));
    // exact GELU
    array gelu = mx::multiply(mx::multiply(m, array(0.5f)),
                              mx::add(array(1.0f), mx::erf(mx::divide(m, array(std::sqrt(2.0f))))));
    m = linear(gelu, get("visual.merger.mlp.2.weight"), get("visual.merger.mlp.2.bias"));
    m = mx::astype(m, mx::float32);
    mx::eval(m);
    const float* dp = m.data<float>();
    return std::vector<float>(dp, dp + static_cast<size_t>(merged) * cfg.hidden_size);
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

  // Multimodal last-token logits: merge vision embeds into the token stream.
  std::vector<float> mm_last_logits(const std::vector<int>& ids,
                                    const std::vector<float>& image_embeds, int n_img,
                                    const std::array<int, 3>& grid) const {
    int L = static_cast<int>(ids.size());
    array h = build_multimodal_embeds(ids, image_embeds, n_img);
    PositionIds pos = get_rope_index(ids, grid);
    h = forward_hidden_from_embeds(h, pos);
    array h_last = mx::take(h, L - 1, 1);
    array logits = mx::astype(linear(h_last, get("model.embed_tokens.weight")), mx::float32);
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

std::vector<float> Qwen2VLModel::forward_vision(const std::vector<float>& pixel_values,
                                                const std::array<int, 3>& grid) const {
  return impl_->forward_vision(pixel_values, grid);
}

std::vector<float> Qwen2VLModel::multimodal_last_logits(const std::vector<int>& input_ids,
                                                        const std::vector<float>& image_embeds,
                                                        int n_img,
                                                        const std::array<int, 3>& grid) const {
  return impl_->mm_last_logits(input_ids, image_embeds, n_img, grid);
}

std::vector<int> Qwen2VLModel::generate_multimodal(const std::vector<int>& input_ids,
                                                   const std::vector<float>& image_embeds, int n_img,
                                                   const std::array<int, 3>& grid, int max_new_tokens,
                                                   const std::vector<int>& eos_ids) const {
  std::vector<int> cur = input_ids;
  std::vector<int> out;
  for (int step = 0; step < max_new_tokens; ++step) {
    std::vector<float> logits = impl_->mm_last_logits(cur, image_embeds, n_img, grid);
    int best = 0;
    for (int i = 1; i < (int)logits.size(); ++i)
      if (logits[i] > logits[best]) best = i;
    bool is_eos = false;
    for (int e : eos_ids) if (best == e) is_eos = true;
    if (is_eos) break;
    out.push_back(best);
    cur.push_back(best);
  }
  return out;
}

}  // namespace mineru

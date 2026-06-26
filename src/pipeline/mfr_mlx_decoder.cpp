// Copyright (c) mlx-mineru.
#include "mineru/mfr_mlx_decoder.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace mineru {
namespace {
namespace mx = mlx::core;
using mx::array;

struct Cfg {
  int d_model = 768, layers = 8, heads = 16, qk_head = 24, v_head = 48;
  int squeeze = 384, ffn = 3072, vocab = 50000, pos_offset = 2, bos = 0, eos = 2;
  float embed_scale = 1.0f, scaling = 0.2041241f;
};

// exact gelu: 0.5 x (1 + erf(x / sqrt(2)))
array gelu(const array& x) {
  return mx::multiply(mx::multiply(x, array(0.5f)),
                      mx::add(array(1.0f), mx::erf(mx::multiply(x, array(0.70710678118f)))));
}
}  // namespace

struct MfrMlxDecoder::Impl {
  std::unordered_map<std::string, array> w;
  Cfg c;

  const array& g(const std::string& n) const {
    auto it = w.find(n);
    if (it == w.end()) throw std::runtime_error("mfr-mlx: missing weight " + n);
    return it->second;
  }
  // y = x @ W + b ; W is [in,out] (ONNX MatMul layout), b is [out].
  array lin(const array& x, const std::string& pfx) const {
    return mx::add(mx::matmul(x, g(pfx + ".weight")), g(pfx + ".bias"));
  }
  array ln(const array& x, const std::string& pfx) const {
    return mx::fast::layer_norm(x, g(pfx + ".weight"), g(pfx + ".bias"), 1e-5f);
  }
};

MfrMlxDecoder::MfrMlxDecoder() : impl_(std::make_unique<Impl>()) {}
MfrMlxDecoder::~MfrMlxDecoder() = default;

std::unique_ptr<MfrMlxDecoder> MfrMlxDecoder::load(const std::string& safetensors_path,
                                                   const std::string& config_json_path) {
  try {
    std::ifstream cf(config_json_path);
    if (!cf) return nullptr;
    nlohmann::json j;
    cf >> j;
    auto r = std::unique_ptr<MfrMlxDecoder>(new MfrMlxDecoder());
    Cfg& c = r->impl_->c;
    c.d_model = j.value("d_model", 768);
    c.layers = j.value("layers", 8);
    c.heads = j.value("heads", 16);
    c.qk_head = j.value("qk_head", 24);
    c.v_head = j.value("v_head", 48);
    c.squeeze = j.value("squeeze", 384);
    c.ffn = j.value("ffn", 3072);
    c.vocab = j.value("vocab", 50000);
    c.pos_offset = j.value("pos_offset", 2);
    c.bos = j.value("bos", 0);
    c.eos = j.value("eos", 2);
    c.embed_scale = j.value("embed_scale", 1.0f);
    c.scaling = j.value("scaling", 1.0f / std::sqrt((float)c.qk_head));
    auto loaded = mx::load_safetensors(safetensors_path);
    if (loaded.first.empty()) return nullptr;
    r->impl_->w = std::move(loaded.first);
    return r;
  } catch (...) {
    return nullptr;
  }
}

std::vector<std::vector<int>> MfrMlxDecoder::decode(const std::vector<std::vector<float>>& enc,
                                                    const std::vector<int>& N, int max_new_tokens,
                                                    int force_steps) const {
  const Impl& m = *impl_;
  const Cfg& c = m.c;
  const int B = (int)enc.size(), D = c.d_model, H = c.heads, QK = c.qk_head, VV = c.v_head;
  std::vector<std::vector<int>> result(B);
  if (B == 0) return result;
  int Nmax = 0;
  for (int n : N) Nmax = std::max(Nmax, n);

  // encoder hidden [B,Nmax,D] zero-padded, + additive cross mask [B,1,1,Nmax].
  std::vector<float> encbuf((size_t)B * Nmax * D, 0.f), maskbuf((size_t)B * Nmax, 0.f);
  for (int b = 0; b < B; ++b) {
    std::copy(enc[b].begin(), enc[b].end(), encbuf.begin() + (size_t)b * Nmax * D);
    for (int j = N[b]; j < Nmax; ++j) maskbuf[(size_t)b * Nmax + j] = -1e9f;
  }
  array encA(encbuf.data(), {B, Nmax, D}, mx::float32);
  array crossMask = mx::reshape(array(maskbuf.data(), {B, Nmax}, mx::float32), {B, 1, 1, Nmax});

  // Cross-attn K/V computed ONCE per layer from the encoder output.
  std::vector<array> ck, cv;
  ck.reserve(c.layers);
  cv.reserve(c.layers);
  for (int l = 0; l < c.layers; ++l) {
    std::string p = "layers." + std::to_string(l) + ".encoder_attn.";
    array k = m.lin(encA, p + "k_proj");  // [B,Nmax,384]
    array v = m.lin(encA, p + "v_proj");  // [B,Nmax,768]
    ck.push_back(mx::transpose(mx::reshape(k, {B, Nmax, H, QK}), {0, 2, 1, 3}));  // [B,H,Nmax,QK]
    cv.push_back(mx::transpose(mx::reshape(v, {B, Nmax, H, VV}), {0, 2, 1, 3}));  // [B,H,Nmax,VV]
  }
  std::vector<array> kc(c.layers, array(0.f)), vc(c.layers, array(0.f));  // self-attn KV cache

  std::vector<int> cur(B, c.bos);
  std::vector<char> done(B, 0);
  int steps = force_steps > 0 ? force_steps : max_new_tokens;
  for (int step = 0; step < steps; ++step) {
    std::vector<int32_t> idsbuf(cur.begin(), cur.end());
    array ids(idsbuf.data(), {B}, mx::int32);
    array emb = mx::multiply(mx::take(m.g("embed_tokens.weight"), ids, 0), array(c.embed_scale));
    int32_t posidx = step + c.pos_offset;
    array pos = mx::take(m.g("embed_positions.weight"), array(&posidx, {1}, mx::int32), 0);  // [1,D]
    array h = mx::reshape(mx::add(emb, pos), {B, 1, D});
    h = m.ln(h, "layernorm_embedding");

    for (int l = 0; l < c.layers; ++l) {
      std::string p = "layers." + std::to_string(l) + ".";
      // ---- self-attention (pre-norm, causal via growing KV cache) ----
      array x = m.ln(h, p + "self_attn_layer_norm");
      array q = mx::multiply(m.lin(x, p + "self_attn.q_proj"), array(c.scaling));
      array k = m.lin(x, p + "self_attn.k_proj");
      array v = m.lin(x, p + "self_attn.v_proj");
      q = mx::transpose(mx::reshape(q, {B, 1, H, QK}), {0, 2, 1, 3});  // [B,H,1,QK]
      k = mx::transpose(mx::reshape(k, {B, 1, H, QK}), {0, 2, 1, 3});
      v = mx::transpose(mx::reshape(v, {B, 1, H, VV}), {0, 2, 1, 3});
      if (step == 0) {
        kc[l] = k;
        vc[l] = v;
      } else {
        kc[l] = mx::concatenate({kc[l], k}, 2);
        vc[l] = mx::concatenate({vc[l], v}, 2);
      }
      array sc = mx::matmul(q, mx::transpose(kc[l], {0, 1, 3, 2}));  // [B,H,1,t+1]
      array o = mx::matmul(mx::softmax(sc, -1), vc[l]);              // [B,H,1,VV]
      o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {B, 1, H * VV});
      h = mx::add(h, m.lin(o, p + "self_attn.out_proj"));
      // ---- cross-attention (pre-norm, cached cross K/V) ----
      x = m.ln(h, p + "encoder_attn_layer_norm");
      q = mx::multiply(m.lin(x, p + "encoder_attn.q_proj"), array(c.scaling));
      q = mx::transpose(mx::reshape(q, {B, 1, H, QK}), {0, 2, 1, 3});
      sc = mx::add(mx::matmul(q, mx::transpose(ck[l], {0, 1, 3, 2})), crossMask);  // [B,H,1,Nmax]
      o = mx::matmul(mx::softmax(sc, -1), cv[l]);
      o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {B, 1, H * VV});
      h = mx::add(h, m.lin(o, p + "encoder_attn.out_proj"));
      // ---- FFN (pre-norm, gelu) ----
      x = m.ln(h, p + "final_layer_norm");
      x = mx::add(mx::matmul(gelu(m.lin(x, p + "fc1")), m.g(p + "fc2.weight")), m.g(p + "fc2.bias"));
      h = mx::add(h, x);
    }
    h = m.ln(h, "layer_norm");
    array logits = mx::reshape(mx::matmul(h, m.g("lm_head.weight")), {B, c.vocab});
    array next = mx::argmax(logits, -1);  // [B] uint32
    mx::eval(next);
    for (int l = 0; l < c.layers; ++l) mx::eval(kc[l], vc[l]);  // realize cache for next step
    const uint32_t* np = next.data<uint32_t>();
    bool all_done = true;
    for (int b = 0; b < B; ++b) {
      int tok = (int)np[b];
      cur[b] = tok;
      if (!done[b]) {
        result[b].push_back(tok);  // include the final EOS, matching the ORT decoder's output
        if (force_steps == 0 && tok == c.eos) done[b] = 1;
      }
      if (!done[b]) all_done = false;
    }
    if (force_steps == 0 && all_done) break;
  }
  return result;
}

}  // namespace mineru

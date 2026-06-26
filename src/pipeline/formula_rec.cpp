// Copyright (c) mlx-mineru.
// UniMERNet formula recognizer: Swin encoder + mBART decoder (greedy, no KV cache) via
// ONNX Runtime, with the UniMERNet grayscale preprocess and a byte-level BPE decode.
#include "mineru/formula_rec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>

#include <filesystem>

#include "mineru/cv_resize.hpp"  // resize_rgb8_cv (real cv2.resize)
#include "mnn_runner.hpp"        // hybrid MNN path (Metal-accelerated encoder + decoder)
#include "onnxruntime_cxx_api.h"
#ifdef MINERU_HAVE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace mineru {
namespace {

constexpr int kTargetH = 192, kTargetW = 672;
constexpr int kBOS = 0, kEOS = 2, kMaxNewTokens = 1536;
const std::set<int> kSpecialIds = {0, 1, 2, 3, 4, 5, 6, 7};

// Merged KV-cache decoder I/O (see scripts/export_mfr_onnx.py): each of the 8 mBART decoder
// layers caches self-attention K (head_dim 24) and V (head_dim 48) across steps; cross-attention
// is recomputed from encoder_hidden each step. Inputs: input_ids[1,1], attention_mask[1,L],
// encoder_hidden_states[1,N,768], self_past_0..15; outputs: logits, self_present_0..15.
constexpr int kDecLayers = 8, kHeads = 16, kKeyDim = 24, kValDim = 48;

// GPT-2 byte<->unicode map: codepoint -> original byte.
std::array<int, 0x200> build_u2b() {
  std::array<int, 0x200> u2b;
  u2b.fill(-1);
  std::vector<int> bs;
  for (int b = 33; b < 127; ++b) bs.push_back(b);
  for (int b = 161; b < 173; ++b) bs.push_back(b);
  for (int b = 174; b < 256; ++b) bs.push_back(b);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) u2b[cs[i]] = bs[i];
  return u2b;
}

// Decode a UTF-8 string into codepoints.
std::vector<uint32_t> utf8_codepoints(const std::string& s) {
  std::vector<uint32_t> out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = s[i];
    uint32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
    else { cp = c & 0x07; len = 4; }
    for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    out.push_back(cp);
    i += len;
  }
  return out;
}

}  // namespace

struct FormulaRecognizer::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-mfr"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> enc, dec;
  std::unique_ptr<MnnRunner> enc_mnn;  // Metal-accelerated encoder when a `.mnn` sibling exists
  std::string enc_in, enc_out;
  std::vector<std::string> vocab;  // id -> token
  std::array<int, 0x200> u2b;
};

FormulaRecognizer::FormulaRecognizer() : impl_(std::make_unique<Impl>()) {}
FormulaRecognizer::~FormulaRecognizer() = default;
FormulaRecognizer::FormulaRecognizer(FormulaRecognizer&&) noexcept = default;
FormulaRecognizer& FormulaRecognizer::operator=(FormulaRecognizer&&) noexcept = default;

FormulaRecognizer FormulaRecognizer::load(const std::string& encoder_onnx,
                                          const std::string& decoder_onnx,
                                          const std::string& vocab_path) {
  FormulaRecognizer r;
  Impl& m = *r.impl_;
  m.u2b = build_u2b();
  std::ifstream vin(vocab_path, std::ios::binary);
  if (!vin) throw std::runtime_error("mfr: cannot open vocab " + vocab_path);
  std::string line;
  while (std::getline(vin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::string tok;  // unescape \\ and \n
    for (size_t i = 0; i < line.size(); ++i) {
      if (line[i] == '\\' && i + 1 < line.size()) {
        char n = line[++i];
        tok += (n == 'n') ? '\n' : n;
      } else {
        tok += line[i];
      }
    }
    m.vocab.push_back(tok);
  }
  m.enc = std::make_unique<Ort::Session>(m.env, encoder_onnx.c_str(), m.opts);
  m.dec = std::make_unique<Ort::Session>(m.env, decoder_onnx.c_str(), m.opts);
  Ort::AllocatorWithDefaultOptions alloc;
  m.enc_in = m.enc->GetInputNameAllocated(0, alloc).get();
  m.enc_out = m.enc->GetOutputNameAllocated(0, alloc).get();
  // The decoder uses the fixed KV-cache I/O names from export_mfr_onnx.py (input_ids,
  // attention_mask, encoder_hidden_states, self_past_*/self_present_*) — no lookup needed.
  //
  // Prefer a `<encoder>.mnn` sibling (Metal-accelerated Swin encoder, ~6.5x; comparable on CPU),
  // falling back to ORT if it's missing or fails to load. The decoder always runs on ORT (the
  // KV cache already makes it fast and backend-independent).
  if (encoder_onnx.size() > 5 && encoder_onnx.substr(encoder_onnx.size() - 5) == ".onnx") {
    std::string mp = encoder_onnx.substr(0, encoder_onnx.size() - 5) + ".mnn";
    std::error_code ec;
    if (std::filesystem::exists(mp, ec)) m.enc_mnn = MnnRunner::load(mp, {m.enc_in}, {m.enc_out});
  }
  return r;
}

FormulaResult FormulaRecognizer::recognize_pixel(const std::vector<float>& gray, int H,
                                                 int W) const {
  const Impl& m = *impl_;
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // [1,3,H,W] with the gray map replicated across channels.
  std::vector<float> px(static_cast<size_t>(3) * H * W);
  for (int c = 0; c < 3; ++c)
    std::copy(gray.begin(), gray.begin() + static_cast<size_t>(H) * W,
              px.begin() + static_cast<size_t>(c) * H * W);
  std::vector<float> hid;
  int N = 0, D = 0;
  // MINERU_DEBUG_MFR=1: report encoder(Metal) vs decoder(ORT) split per formula.
  const bool dbg_mfr = [] { const char* e = std::getenv("MINERU_DEBUG_MFR"); return e && *e && *e != '0'; }();
  auto _t_enc0 = std::chrono::steady_clock::now();
  if (m.enc_mnn) {  // MNN/Metal encoder
    std::vector<std::vector<float>> eo;
    std::vector<std::vector<int>> es;
    if (!m.enc_mnn->run(px.data(), {1, 3, H, W}, eo, es) || eo.empty() || es[0].size() < 3)
      throw std::runtime_error("mfr: MNN encoder failed");
    N = es[0][1];
    D = es[0][2];
    hid = std::move(eo[0]);
  } else {  // ONNX Runtime encoder
    std::array<int64_t, 4> pshape{1, 3, H, W};
    Ort::Value pv = Ort::Value::CreateTensor<float>(mi, px.data(), px.size(), pshape.data(), 4);
    const char* ein[] = {m.enc_in.c_str()};
    const char* eout[] = {m.enc_out.c_str()};
    auto enc_outs =
        const_cast<Ort::Session&>(*m.enc).Run(Ort::RunOptions{nullptr}, ein, &pv, 1, eout, 1);
    auto hshape = enc_outs[0].GetTensorTypeAndShapeInfo().GetShape();
    N = static_cast<int>(hshape[1]);
    D = static_cast<int>(hshape[2]);
    const float* hsrc = enc_outs[0].GetTensorData<float>();
    hid.assign(hsrc, hsrc + static_cast<size_t>(N) * D);
  }
  std::array<int64_t, 3> hidshape{1, N, D};
  double enc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_enc0).count();
  auto _t_dec0 = std::chrono::steady_clock::now();

  // Greedy decode with a KV cache: the merged decoder graph caches each layer's self-attention
  // K/V (fed back present->past every step) so a step processes only the new token, instead of
  // recomputing the whole prefix (O(T) instead of O(T^2)). Cross-attention is recomputed from
  // encoder_hidden each step (cheap, O(N)). See scripts/export_mfr_onnx.py.
  static const std::vector<std::string> kInNamesS = [] {
    std::vector<std::string> v{"input_ids", "attention_mask", "encoder_hidden_states"};
    for (int i = 0; i < 2 * kDecLayers; ++i) v.push_back("self_past_" + std::to_string(i));
    return v;
  }();
  static const std::vector<std::string> kOutNamesS = [] {
    std::vector<std::string> v{"logits"};
    for (int i = 0; i < 2 * kDecLayers; ++i) v.push_back("self_present_" + std::to_string(i));
    return v;
  }();
  std::vector<const char*> in_names, out_names;
  for (auto& s : kInNamesS) in_names.push_back(s.c_str());
  for (auto& s : kOutNamesS) out_names.push_back(s.c_str());

  // Cache tensors, fed as self_past_* and replaced by self_present_* each step. Start empty
  // (self-attn sequence length 0) so the first call is the prefill.
  std::vector<Ort::Value> cache;
  cache.reserve(2 * kDecLayers);
  std::vector<std::vector<float>> empty(2 * kDecLayers);  // backs the seq-0 prefill tensors
  for (int i = 0; i < 2 * kDecLayers; ++i) {
    int64_t dim = (i % 2 == 0) ? kKeyDim : kValDim;
    std::array<int64_t, 4> sh{1, kHeads, 0, dim};
    cache.push_back(Ort::Value::CreateTensor<float>(mi, empty[i].data(), 0, sh.data(), 4));
  }

  FormulaResult res;
  int64_t cur = kBOS;
  for (int step = 0; step < kMaxNewTokens; ++step) {
    int64_t id_val = cur;
    std::array<int64_t, 2> ishape{1, 1};
    std::vector<int64_t> mask(step + 1, 1);  // attention over all tokens so far (length = step+1)
    std::array<int64_t, 2> mshape{1, step + 1};

    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + 2 * kDecLayers);
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mi, &id_val, 1, ishape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mi, mask.data(), mask.size(), mshape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<float>(mi, hid.data(), hid.size(), hidshape.data(), 3));
    for (auto& c : cache) inputs.push_back(std::move(c));  // self_past_* (consumed this step)

    auto outs = const_cast<Ort::Session&>(*m.dec).Run(
        Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(),
        out_names.size());

    auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    int V = static_cast<int>(oshape.back());  // logits [1,1,V]
    const float* logits = outs[0].GetTensorData<float>();
    int best = 0;
    float bv = logits[0];
    for (int v = 1; v < V; ++v)
      if (logits[v] > bv) { bv = logits[v]; best = v; }
    res.ids.push_back(best);
    if (best == kEOS) break;
    cur = best;
    cache.clear();  // present -> past for the next step (no copy; reuse ORT-owned buffers)
    for (int i = 0; i < 2 * kDecLayers; ++i) cache.push_back(std::move(outs[1 + i]));
  }

  if (dbg_mfr) {
    double dec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t_dec0).count();
    std::fprintf(stderr, "[mfr-debug] encoder(%s)=%.1f ms | decoder(ORT, %d steps)=%.1f ms\n",
                 m.enc_mnn ? "MNN/Metal" : "ORT", enc_ms, (int)res.ids.size(), dec_ms);
  }
  // Byte-level decode (skip special tokens).
  std::string bytes;
  for (int id : res.ids) {
    if (kSpecialIds.count(id) || id < 0 || id >= (int)m.vocab.size()) continue;
    for (uint32_t cp : utf8_codepoints(m.vocab[id])) {
      int b = (cp < m.u2b.size()) ? m.u2b[cp] : -1;
      if (b >= 0) bytes += static_cast<char>(b);
    }
  }
  res.latex = bytes;
  return res;
}

FormulaResult FormulaRecognizer::recognize(const std::vector<uint8_t>& rgb, int w, int h) const {
#ifdef MINERU_HAVE_OPENCV
  // Faithful UniMERNet preprocess via real cv2 (crop_margin_numpy -> aspect resize ->
  // center pad -> gray normalize), so it matches MinerU byte-for-byte.
  cv::Mat img(h, w, CV_8UC3, const_cast<uint8_t*>(rgb.data()));  // RGB
  // crop_margin_numpy: gray, min-max stretch, threshold <200, boundingRect of content.
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
  double gmin, gmax;
  cv::minMaxLoc(gray, &gmin, &gmax);
  cv::Mat crop = img;
  if (gmax != gmin) {
    cv::Mat norm;
    gray.convertTo(norm, CV_64F);
    norm = (norm - gmin) / (gmax - gmin) * 255.0;
    // numpy .astype(uint8) TRUNCATES (not rounds); for the <200 threshold floor(n)<200 <=> n<200,
    // so threshold the float directly (cv::convertTo would round and shift the edge by 0.5).
    cv::Mat binary = (norm < 200.0);  // 0/255
    std::vector<cv::Point> nz;
    cv::findNonZero(binary, nz);
    if (!nz.empty()) {
      cv::Rect r = cv::boundingRect(nz);
      crop = img(r);
    }
  }
  int cw = crop.cols, ch = crop.rows;
  double scale = std::min((double)kTargetH / ch, (double)kTargetW / cw);
  int nh = std::max(1, (int)(ch * scale)), nw = std::max(1, (int)(cw * scale));
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(nw, nh));
  // center pad (copyMakeBorder, BORDER_CONSTANT black) to 192x672.
  int pad_w = (kTargetW - nw) / 2, pad_h = (kTargetH - nh) / 2;
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, pad_h, kTargetH - nh - pad_h, pad_w, kTargetW - nw - pad_w,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  // to_normalized_gray_tensor: RGB2GRAY + (g - 0.7931*255)/(0.1738*255).
  cv::Mat pg;
  cv::cvtColor(padded, pg, cv::COLOR_RGB2GRAY);
  std::vector<float> px(static_cast<size_t>(kTargetH) * kTargetW);
  for (int y = 0; y < kTargetH; ++y)
    for (int x = 0; x < kTargetW; ++x)
      px[(size_t)y * kTargetW + x] = (pg.at<uint8_t>(y, x) - 0.7931f * 255.0f) / (0.1738f * 255.0f);
  return recognize_pixel(px, kTargetH, kTargetW);
#else
  (void)rgb; (void)w; (void)h;
  return {};
#endif
}

}  // namespace mineru

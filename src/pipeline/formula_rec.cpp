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
#ifdef MINERU_HAVE_MLX
#include "mineru/mfr_mlx_decoder.hpp"  // MLX/Metal batched decoder (validated == ORT)
#endif
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
#ifdef MINERU_HAVE_MLX
  std::unique_ptr<MfrMlxDecoder> mlx_dec;  // MLX/Metal decoder when its safetensors are present
#endif
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
  std::error_code ec;
  // Encoder: prefer the `<encoder>.mnn` sibling (Metal-accelerated Swin, ~6.5x). Its I/O names are
  // fixed (export_mfr_onnx.py: pixel_values -> encoder_hidden), so we can load MNN WITHOUT first
  // creating the ORT session. Only build the ORT encoder as a fallback when the .mnn is absent or
  // fails — that avoids loading the ~235MB encoder .onnx whenever the .mnn is in use.
  m.enc_in = "pixel_values";
  m.enc_out = "encoder_hidden";
  if (encoder_onnx.size() > 5 && encoder_onnx.substr(encoder_onnx.size() - 5) == ".onnx") {
    std::string mp = encoder_onnx.substr(0, encoder_onnx.size() - 5) + ".mnn";
    if (std::filesystem::exists(mp, ec)) m.enc_mnn = MnnRunner::load(mp, {m.enc_in}, {m.enc_out});
  }
  if (!m.enc_mnn) {  // fallback: ORT encoder
    m.enc = std::make_unique<Ort::Session>(m.env, encoder_onnx.c_str(), m.opts);
    Ort::AllocatorWithDefaultOptions alloc;
    m.enc_in = m.enc->GetInputNameAllocated(0, alloc).get();
    m.enc_out = m.enc->GetOutputNameAllocated(0, alloc).get();
  }
#ifdef MINERU_HAVE_MLX
  // Decoder: prefer the MLX/Metal decoder (mfr_decoder.safetensors + _config.json from
  // scripts/extract_mfr_decoder_weights.py; byte-identical to ORT, opt out with MINERU_MFR_MLX=0).
  const char* mlx_off = std::getenv("MINERU_MFR_MLX");
  if (!(mlx_off && std::string(mlx_off) == "0") &&
      decoder_onnx.size() > 5 && decoder_onnx.substr(decoder_onnx.size() - 5) == ".onnx") {
    std::string base = decoder_onnx.substr(0, decoder_onnx.size() - 5);
    if (std::filesystem::exists(base + ".safetensors", ec))
      m.mlx_dec = MfrMlxDecoder::load(base + ".safetensors", base + "_config.json");
  }
#endif
  // ORT decoder: only as the fallback when the MLX decoder isn't in use (avoids loading the
  // ~577MB merged KV-cache .onnx whenever MLX runs). Uses the fixed KV-cache I/O names.
  bool have_mlx = false;
#ifdef MINERU_HAVE_MLX
  have_mlx = (m.mlx_dec != nullptr);
#endif
  if (!have_mlx) m.dec = std::make_unique<Ort::Session>(m.env, decoder_onnx.c_str(), m.opts);
  return r;
}

namespace {
const bool g_dbg_mfr = [] { const char* e = std::getenv("MINERU_DEBUG_MFR"); return e && *e && *e != '0'; }();

// Run the Swin encoder (MNN/Metal or ORT) on a preprocessed gray map -> hidden state.
FormulaEncoded run_encoder(const FormulaRecognizer::Impl& m, const std::vector<float>& gray,
                           int H, int W);
// Single-formula greedy decode on the ORT KV-cache graph -> token ids.
std::vector<int> decode_ort(const FormulaRecognizer::Impl& m, const std::vector<float>& hid, int N);
}  // namespace

// Byte-level decode (skip special tokens): token ids -> LaTeX. Defined after Impl is complete.
static std::string ids_to_latex(const FormulaRecognizer::Impl& m, const std::vector<int>& ids);

FormulaResult FormulaRecognizer::recognize_pixel(const std::vector<float>& gray, int H,
                                                 int W) const {
  return decode_batch({run_encoder(*impl_, gray, H, W)}).at(0);
}

std::vector<FormulaResult> FormulaRecognizer::decode_batch(
    const std::vector<FormulaEncoded>& encs) const {
  const Impl& m = *impl_;
  std::vector<FormulaResult> out(encs.size());
  if (encs.empty()) return out;
  auto _t0 = std::chrono::steady_clock::now();
#ifdef MINERU_HAVE_MLX
  if (m.mlx_dec) {  // ONE batched GPU decode over all formulas
    std::vector<std::vector<float>> hids;
    std::vector<int> Ns;
    hids.reserve(encs.size());
    for (const auto& e : encs) { hids.push_back(e.hid); Ns.push_back(e.n); }
    auto ids = m.mlx_dec->decode(hids, Ns, kMaxNewTokens);
    long steps = 0;
    for (size_t i = 0; i < encs.size(); ++i) {
      out[i].ids = std::move(ids[i]);
      out[i].latex = ids_to_latex(m, out[i].ids);
      steps = std::max(steps, (long)out[i].ids.size());
    }
    if (g_dbg_mfr) {
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
      std::fprintf(stderr, "[mfr-debug] MLX/Metal batched decode: %zu formula(s), %ld steps, %.1f ms\n",
                   encs.size(), steps, ms);
    }
    return out;
  }
#endif
  for (size_t i = 0; i < encs.size(); ++i) {  // ORT: one formula at a time
    out[i].ids = decode_ort(m, encs[i].hid, encs[i].n);
    out[i].latex = ids_to_latex(m, out[i].ids);
  }
  if (g_dbg_mfr) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
    std::fprintf(stderr, "[mfr-debug] ORT decode: %zu formula(s), %.1f ms\n", encs.size(), ms);
  }
  return out;
}

static std::string ids_to_latex(const FormulaRecognizer::Impl& m, const std::vector<int>& ids) {
  std::string bytes;
  for (int id : ids) {
    if (kSpecialIds.count(id) || id < 0 || id >= (int)m.vocab.size()) continue;
    for (uint32_t cp : utf8_codepoints(m.vocab[id])) {
      int b = (cp < m.u2b.size()) ? m.u2b[cp] : -1;
      if (b >= 0) bytes += static_cast<char>(b);
    }
  }
  return bytes;
}

namespace {
FormulaEncoded run_encoder(const FormulaRecognizer::Impl& m, const std::vector<float>& gray,
                           int H, int W) {
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> px(static_cast<size_t>(3) * H * W);  // [1,3,H,W], gray replicated to 3 ch
  for (int c = 0; c < 3; ++c)
    std::copy(gray.begin(), gray.begin() + static_cast<size_t>(H) * W,
              px.begin() + static_cast<size_t>(c) * H * W);
  FormulaEncoded e;
  auto _t = std::chrono::steady_clock::now();
  if (m.enc_mnn) {  // MNN/Metal encoder
    std::vector<std::vector<float>> eo;
    std::vector<std::vector<int>> es;
    if (!m.enc_mnn->run(px.data(), {1, 3, H, W}, eo, es) || eo.empty() || es[0].size() < 3)
      throw std::runtime_error("mfr: MNN encoder failed");
    e.n = es[0][1];
    e.hid = std::move(eo[0]);
  } else {  // ONNX Runtime encoder
    std::array<int64_t, 4> pshape{1, 3, H, W};
    Ort::Value pv = Ort::Value::CreateTensor<float>(mi, px.data(), px.size(), pshape.data(), 4);
    const char* ein[] = {m.enc_in.c_str()};
    const char* eout[] = {m.enc_out.c_str()};
    auto enc_outs =
        const_cast<Ort::Session&>(*m.enc).Run(Ort::RunOptions{nullptr}, ein, &pv, 1, eout, 1);
    auto hshape = enc_outs[0].GetTensorTypeAndShapeInfo().GetShape();
    e.n = static_cast<int>(hshape[1]);
    int D = static_cast<int>(hshape[2]);
    const float* hsrc = enc_outs[0].GetTensorData<float>();
    e.hid.assign(hsrc, hsrc + static_cast<size_t>(e.n) * D);
  }
  if (g_dbg_mfr) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t).count();
    std::fprintf(stderr, "[mfr-debug] encoder(%s)=%.1f ms (N=%d)\n", m.enc_mnn ? "MNN/Metal" : "ORT", ms, e.n);
  }
  return e;
}

std::vector<int> decode_ort(const FormulaRecognizer::Impl& m, const std::vector<float>& hid, int N) {
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  int D = N > 0 ? (int)(hid.size() / N) : 0;
  std::array<int64_t, 3> hidshape{1, N, D};
  // Greedy decode with a KV cache: the merged decoder graph caches each layer's self-attention
  // K/V (present->past each step) so a step processes only the new token (O(T) not O(T^2)).
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

  std::vector<Ort::Value> cache;
  cache.reserve(2 * kDecLayers);
  std::vector<std::vector<float>> empty(2 * kDecLayers);
  for (int i = 0; i < 2 * kDecLayers; ++i) {
    int64_t dim = (i % 2 == 0) ? kKeyDim : kValDim;
    std::array<int64_t, 4> sh{1, kHeads, 0, dim};
    cache.push_back(Ort::Value::CreateTensor<float>(mi, empty[i].data(), 0, sh.data(), 4));
  }
  std::vector<int> ids;
  int64_t cur = kBOS;
  for (int step = 0; step < kMaxNewTokens; ++step) {
    int64_t id_val = cur;
    std::array<int64_t, 2> ishape{1, 1};
    std::vector<int64_t> mask(step + 1, 1);
    std::array<int64_t, 2> mshape{1, step + 1};
    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + 2 * kDecLayers);
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mi, &id_val, 1, ishape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mi, mask.data(), mask.size(), mshape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<float>(mi, const_cast<float*>(hid.data()), hid.size(), hidshape.data(), 3));
    for (auto& c : cache) inputs.push_back(std::move(c));
    auto outs = const_cast<Ort::Session&>(*m.dec).Run(
        Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(), out_names.data(),
        out_names.size());
    int V = static_cast<int>(outs[0].GetTensorTypeAndShapeInfo().GetShape().back());
    const float* logits = outs[0].GetTensorData<float>();
    int best = 0;
    float bv = logits[0];
    for (int v = 1; v < V; ++v)
      if (logits[v] > bv) { bv = logits[v]; best = v; }
    ids.push_back(best);
    if (best == kEOS) break;
    cur = best;
    cache.clear();
    for (int i = 0; i < 2 * kDecLayers; ++i) cache.push_back(std::move(outs[1 + i]));
  }
  return ids;
}
}  // namespace

// UniMERNet preprocess (crop-margin -> aspect resize -> center pad -> gray normalize) -> the
// [kTargetH*kTargetW] gray map the encoder consumes. Empty when built without OpenCV.
static std::vector<float> preprocess_to_gray(const std::vector<uint8_t>& rgb, int w, int h) {
#ifdef MINERU_HAVE_OPENCV
  // Faithful UniMERNet preprocess via real cv2, so it matches MinerU byte-for-byte.
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
  return px;
#else
  (void)rgb; (void)w; (void)h;
  return {};
#endif
}

FormulaResult FormulaRecognizer::recognize(const std::vector<uint8_t>& rgb, int w, int h) const {
  std::vector<float> px = preprocess_to_gray(rgb, w, h);
  if (px.empty()) return {};
  return recognize_pixel(px, kTargetH, kTargetW);
}

FormulaEncoded FormulaRecognizer::encode(const std::vector<uint8_t>& rgb, int w, int h) const {
  std::vector<float> px = preprocess_to_gray(rgb, w, h);
  if (px.empty()) return {};
  return run_encoder(*impl_, px, kTargetH, kTargetW);
}

}  // namespace mineru

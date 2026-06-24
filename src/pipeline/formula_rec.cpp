// Copyright (c) mlx-mineru.
// UniMERNet formula recognizer: Swin encoder + mBART decoder (greedy, no KV cache) via
// ONNX Runtime, with the UniMERNet grayscale preprocess and a byte-level BPE decode.
#include "mineru/formula_rec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

#include "mineru/image_preprocess.hpp"  // resize_bilinear_rgb8
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {

constexpr int kTargetH = 192, kTargetW = 672;
constexpr int kBOS = 0, kEOS = 2, kMaxNewTokens = 1536;
const std::set<int> kSpecialIds = {0, 1, 2, 3, 4, 5, 6, 7};

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
  std::string enc_in, enc_out, dec_in_ids, dec_in_hid, dec_out;
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
  m.dec_in_ids = m.dec->GetInputNameAllocated(0, alloc).get();
  m.dec_in_hid = m.dec->GetInputNameAllocated(1, alloc).get();
  m.dec_out = m.dec->GetOutputNameAllocated(0, alloc).get();
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
  std::array<int64_t, 4> pshape{1, 3, H, W};
  Ort::Value pv = Ort::Value::CreateTensor<float>(mi, px.data(), px.size(), pshape.data(), 4);
  const char* ein[] = {m.enc_in.c_str()};
  const char* eout[] = {m.enc_out.c_str()};
  auto enc_outs =
      const_cast<Ort::Session&>(*m.enc).Run(Ort::RunOptions{nullptr}, ein, &pv, 1, eout, 1);
  auto hshape = enc_outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int N = static_cast<int>(hshape[1]), D = static_cast<int>(hshape[2]);
  const float* hsrc = enc_outs[0].GetTensorData<float>();
  std::vector<float> hid(hsrc, hsrc + static_cast<size_t>(N) * D);
  std::array<int64_t, 3> hidshape{1, N, D};

  // Greedy decode (recompute full sequence each step).
  FormulaResult res;
  std::vector<int64_t> seq{kBOS};
  const char* din[] = {m.dec_in_ids.c_str(), m.dec_in_hid.c_str()};
  const char* dout[] = {m.dec_out.c_str()};
  for (int step = 0; step < kMaxNewTokens; ++step) {
    std::array<int64_t, 2> ishape{1, (int64_t)seq.size()};
    Ort::Value ids =
        Ort::Value::CreateTensor<int64_t>(mi, seq.data(), seq.size(), ishape.data(), 2);
    Ort::Value hv =
        Ort::Value::CreateTensor<float>(mi, hid.data(), hid.size(), hidshape.data(), 3);
    std::array<Ort::Value, 2> ins{std::move(ids), std::move(hv)};
    auto outs = const_cast<Ort::Session&>(*m.dec).Run(Ort::RunOptions{nullptr}, din, ins.data(),
                                                      2, dout, 1);
    auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    int T = static_cast<int>(oshape[1]), V = static_cast<int>(oshape[2]);
    const float* logits = outs[0].GetTensorData<float>() + static_cast<size_t>(T - 1) * V;
    int best = 0;
    float bv = logits[0];
    for (int v = 1; v < V; ++v)
      if (logits[v] > bv) { bv = logits[v]; best = v; }
    res.ids.push_back(best);
    seq.push_back(best);
    if (best == kEOS) break;
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
  // --- UniMERNet preprocess: crop_margin -> aspect resize -> center pad -> gray normalize ---
  // crop_margin_numpy: grayscale, min-max stretch, threshold <200, bounding box of content.
  auto gray_at = [&](int x, int y) {
    const uint8_t* p = &rgb[(static_cast<size_t>(y) * w + x) * 3];
    return (p[0] * 4899 + p[1] * 9617 + p[2] * 1868 + 8192) >> 14;  // cv2 RGB2GRAY
  };
  int gmin = 255, gmax = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int g = gray_at(x, y);
      gmin = std::min(gmin, g);
      gmax = std::max(gmax, g);
    }
  int cx0 = 0, cy0 = 0, cw = w, ch = h;
  if (gmax > gmin) {
    int xmin = w, ymin = h, xmax = -1, ymax = -1;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        int norm = ((gray_at(x, y) - gmin) * 255) / (gmax - gmin);
        if (norm < 200) {  // content pixel
          xmin = std::min(xmin, x); ymin = std::min(ymin, y);
          xmax = std::max(xmax, x); ymax = std::max(ymax, y);
        }
      }
    if (xmax >= xmin && ymax >= ymin) {
      cx0 = xmin; cy0 = ymin; cw = xmax - xmin + 1; ch = ymax - ymin + 1;
    }
  }
  std::vector<uint8_t> crop(static_cast<size_t>(cw) * ch * 3);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c)
        crop[(static_cast<size_t>(y) * cw + x) * 3 + c] =
            rgb[(static_cast<size_t>(cy0 + y) * w + (cx0 + x)) * 3 + c];

  double scale = std::min((double)kTargetH / ch, (double)kTargetW / cw);
  int nh = (int)(ch * scale), nw = (int)(cw * scale);
  nh = std::max(1, nh);
  nw = std::max(1, nw);
  std::vector<uint8_t> resized = resize_bilinear_rgb8(crop, cw, ch, nw, nh);

  int pad_w = (kTargetW - nw) / 2, pad_h = (kTargetH - nh) / 2;
  // gray-normalize directly into the padded canvas (black padding -> gray 0).
  std::vector<float> px(static_cast<size_t>(kTargetH) * kTargetW, (0.0f - 0.7931f * 255.0f) / (0.1738f * 255.0f));
  for (int y = 0; y < nh; ++y)
    for (int x = 0; x < nw; ++x) {
      const uint8_t* p = &resized[(static_cast<size_t>(y) * nw + x) * 3];
      int g = (p[0] * 4899 + p[1] * 9617 + p[2] * 1868 + 8192) >> 14;
      px[static_cast<size_t>(pad_h + y) * kTargetW + (pad_w + x)] =
          (g - 0.7931f * 255.0f) / (0.1738f * 255.0f);
    }
  return recognize_pixel(px, kTargetH, kTargetW);
}

}  // namespace mineru

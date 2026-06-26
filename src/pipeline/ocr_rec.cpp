// Copyright (c) mlx-mineru.
#include "mineru/ocr_rec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "mineru/cv_resize.hpp"  // resize_rgb8_cv (real cv2.resize)
#include "mnn_runner.hpp"        // hybrid MNN path (faster; parity-verified)
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {
constexpr int kH = 48, kWbase = 320, kMinW = 16, kMaxW = 2560;
constexpr int kBatchPad = 6;  // pad batch to a multiple of this so the shape is stable (= rec_batch_num)
}  // namespace

struct TextRecognizer::Impl {
  std::vector<std::string> character;  // [0]="blank", 1..N-2=dict, [N-1]=" "
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-rec"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;  // ORT path (fallback)
  std::unique_ptr<MnnRunner> mnn;         // MNN path (preferred when a .mnn sibling exists)
  std::string in_name, out_name;
};

TextRecognizer::TextRecognizer() : impl_(std::make_unique<Impl>()) {}
TextRecognizer::~TextRecognizer() = default;
TextRecognizer::TextRecognizer(TextRecognizer&&) noexcept = default;
TextRecognizer& TextRecognizer::operator=(TextRecognizer&&) noexcept = default;

TextRecognizer TextRecognizer::load(const std::string& onnx_path, const std::string& dict_path) {
  TextRecognizer r;
  Impl& m = *r.impl_;
  m.character.push_back("blank");  // CTC blank at index 0
  std::ifstream din(dict_path, std::ios::binary);
  if (!din) throw std::runtime_error("ocr_rec: cannot open dict " + dict_path);
  std::string line;
  while (std::getline(din, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty() && line.back() == '\n') line.pop_back();
    m.character.push_back(line);
  }
  m.character.push_back(" ");  // use_space_char
  if (onnx_path.size() > 5 && onnx_path.substr(onnx_path.size() - 5) == ".onnx") {
    std::string mp = onnx_path.substr(0, onnx_path.size() - 5) + ".mnn";
    std::error_code ec;
    // fp16 on Metal (~15-25% faster): SVTR output is CTC-argmax, golden-verified at half precision.
    if (std::filesystem::exists(mp, ec)) m.mnn = MnnRunner::load(mp, {"x"}, {"y"}, /*fp16=*/true);
  }
  if (!m.mnn) {
    m.session = std::make_unique<Ort::Session>(m.env, onnx_path.c_str(), m.opts);
    Ort::AllocatorWithDefaultOptions alloc;
    m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
    m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  }
  if (m.mnn) {
    // Pre-compile the common batched shapes (MNN/Metal recompiles per shape; doing it once here
    // means the first table doesn't pay the shader-compile and the batch win shows immediately).
    for (int W : {kWbase, 2 * kWbase}) {
      std::vector<float> dummy(static_cast<size_t>(kBatchPad) * 3 * kH * W, 0.0f);
      std::vector<std::vector<float>> o;
      std::vector<std::vector<int>> s;
      m.mnn->run(dummy.data(), {kBatchPad, 3, kH, W}, o, s);
    }
  }
  return r;
}

namespace {
// CTC greedy decode of a [T,C] logit map: argmax per step, drop blank(0), collapse repeats;
// score = mean per-step softmax confidence of the kept characters.
RecResult ctc_decode(const float* logits, int T, int C, const std::vector<std::string>& character) {
  RecResult res;
  res.score = 0.0f;
  std::string text;
  int kept = 0, prev = -1;
  for (int t = 0; t < T; ++t) {
    const float* row = logits + static_cast<size_t>(t) * C;
    int best = 0;
    float bestv = row[0];
    for (int c = 1; c < C; ++c)
      if (row[c] > bestv) { bestv = row[c]; best = c; }
    if (best != 0 && best != prev) {
      float mx = bestv, sum = 0.0f;
      for (int c = 0; c < C; ++c) sum += std::exp(row[c] - mx);
      res.score += 1.0f / sum;  // exp(max-max)/sum
      ++kept;
      if (best < (int)character.size()) text += character[best];
    }
    prev = best;
  }
  res.text = text;
  res.score = kept ? res.score / kept : 0.0f;
  return res;
}

// Fill one batch row: resize crop to its width (capped at imgW) and write BGR-normalized into dst.
void fill_rec_row(float* dst, const std::vector<uint8_t>& rgb, int w, int h, int imgW) {
  if (w <= 0 || h <= 0) return;  // degenerate -> all-zero row
  double ratio = (double)w / h;
  int rw = std::min(imgW, std::max((int)std::ceil(kH * ratio), kMinW));
  std::vector<uint8_t> resized = resize_rgb8_cv(rgb, w, h, rw, kH, kInterLinear);
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < rw; ++x)
      for (int c = 0; c < 3; ++c)  // RGB source -> BGR model channel (2-c), /127.5-1
        dst[(static_cast<size_t>(c) * kH + y) * imgW + x] =
            resized[(static_cast<size_t>(y) * rw + x) * 3 + (2 - c)] / 127.5f - 1.0f;
}
}  // namespace

std::vector<RecResult> TextRecognizer::recognize_batch(
    const std::vector<const std::vector<uint8_t>*>& rgbs, const std::vector<int>& ws,
    const std::vector<int>& hs, double max_wh) const {
  const Impl& m = *impl_;
  const int B = (int)rgbs.size();
  std::vector<RecResult> out(B);
  if (B == 0) return out;
  // Shared width, bucketed UP to a multiple of kWbase so only a few distinct shapes ever reach
  // MNN (each compiled once). Extra width is blank padding -> CTC ignores it (same decoded text).
  double mwh = std::max(max_wh, (double)kWbase / kH);
  int rawW = (int)(kH * mwh);
  int imgW = std::min(kMaxW, ((rawW + kWbase - 1) / kWbase) * kWbase);
  imgW = std::max(imgW, kMinW);
  // Pad the batch to a multiple of kBatchPad so the batch dim is also stable.
  int Bp = ((B + kBatchPad - 1) / kBatchPad) * kBatchPad;
  const size_t per = static_cast<size_t>(3) * kH * imgW;
  std::vector<float> input(per * Bp, 0.0f);
  for (int b = 0; b < B; ++b) fill_rec_row(input.data() + per * b, *rgbs[b], ws[b], hs[b], imgW);

  std::vector<float> logits_buf;
  int T, C;
  if (m.mnn) {
    std::vector<std::vector<float>> mouts;
    std::vector<std::vector<int>> moshs;
    if (!m.mnn->run(input.data(), {Bp, 3, kH, imgW}, mouts, moshs) || mouts.empty() ||
        moshs[0].size() < 3)
      throw std::runtime_error("ocr_rec: MNN batch inference failed");
    T = moshs[0][1];
    C = moshs[0][2];
    logits_buf = std::move(mouts[0]);
  } else {
    std::array<int64_t, 4> ishape{Bp, 3, kH, imgW};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
    const char* in_names[] = {m.in_name.c_str()};
    const char* out_names[] = {m.out_name.c_str()};
    auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                          out_names, 1);
    const float* p = outs[0].GetTensorData<float>();
    auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    T = static_cast<int>(oshape[1]);
    C = static_cast<int>(oshape[2]);
    logits_buf.assign(p, p + static_cast<size_t>(Bp) * T * C);
  }
  for (int b = 0; b < B; ++b)
    out[b] = ctc_decode(logits_buf.data() + static_cast<size_t>(b) * T * C, T, C, m.character);
  return out;
}

RecResult TextRecognizer::recognize(const std::vector<uint8_t>& rgb, int w, int h,
                                    double max_wh_override) const {
  const Impl& m = *impl_;
  // Width per MinerU resize_norm_img. max_wh_ratio = batch-shared aspect (or w/h for a
  // single crop), then max'd with imgW/imgH; imgW = int(imgH * max_wh_ratio).
  double max_wh = max_wh_override >= 0.0 ? std::max(max_wh_override, (double)kWbase / kH)
                                        : std::max((double)w / h, (double)kWbase / kH);
  int imgW = std::max(kMinW, std::min(kMaxW, (int)(kH * max_wh)));
  double ratio = (double)w / h;
  int ratio_imgH = std::max((int)std::ceil(kH * ratio), kMinW);
  int rw = std::min(imgW, ratio_imgH);

  std::vector<uint8_t> resized = resize_rgb8_cv(rgb, w, h, rw, kH, kInterLinear);
  std::vector<float> input(static_cast<size_t>(3) * kH * imgW, 0.0f);  // zero-padded
  // MinerU recognizes from BGR crops (cv2). Source is RGB -> model channel c reads (2-c).
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < rw; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * kH + y) * imgW + x] =
            resized[(static_cast<size_t>(y) * rw + x) * 3 + (2 - c)] / 127.5f - 1.0f;

  std::vector<float> logits_buf;
  int T, C;
  if (m.mnn) {
    std::vector<std::vector<float>> mouts;
    std::vector<std::vector<int>> moshs;
    if (!m.mnn->run(input.data(), {1, 3, kH, imgW}, mouts, moshs) || mouts.empty() ||
        moshs[0].size() < 3)
      throw std::runtime_error("ocr_rec: MNN inference failed");
    T = moshs[0][1];
    C = moshs[0][2];
    logits_buf = std::move(mouts[0]);
  } else {
    std::array<int64_t, 4> ishape{1, 3, kH, imgW};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
    const char* in_names[] = {m.in_name.c_str()};
    const char* out_names[] = {m.out_name.c_str()};
    auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                          out_names, 1);
    const float* p = outs[0].GetTensorData<float>();
    auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    T = static_cast<int>(oshape[1]);
    C = static_cast<int>(oshape[2]);
    logits_buf.assign(p, p + static_cast<size_t>(T) * C);
  }
  return ctc_decode(logits_buf.data(), T, C, m.character);
}

}  // namespace mineru

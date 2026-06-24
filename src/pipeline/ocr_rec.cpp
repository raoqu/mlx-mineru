// Copyright (c) mlx-mineru.
#include "mineru/ocr_rec.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "mineru/image_preprocess.hpp"  // resize_bilinear_rgb8
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {
constexpr int kH = 48, kWbase = 320, kMinW = 16, kMaxW = 2560;
}  // namespace

struct TextRecognizer::Impl {
  std::vector<std::string> character;  // [0]="blank", 1..N-2=dict, [N-1]=" "
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-rec"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
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
  m.session = std::make_unique<Ort::Session>(m.env, onnx_path.c_str(), m.opts);
  Ort::AllocatorWithDefaultOptions alloc;
  m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
  m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  return r;
}

RecResult TextRecognizer::recognize(const std::vector<uint8_t>& rgb, int w, int h) const {
  const Impl& m = *impl_;
  // Width per MinerU resize_norm_img (single crop, max_wh_ratio >= imgW/imgH).
  double max_wh = std::max((double)w / h, (double)kWbase / kH);
  int imgW = std::max(kMinW, std::min(kMaxW, (int)(kH * max_wh)));
  double ratio = (double)w / h;
  int ratio_imgH = std::max((int)std::ceil(kH * ratio), kMinW);
  int rw = std::min(imgW, ratio_imgH);

  std::vector<uint8_t> resized = resize_bilinear_rgb8(rgb, w, h, rw, kH);
  std::vector<float> input(static_cast<size_t>(3) * kH * imgW, 0.0f);  // zero-padded
  // MinerU recognizes from BGR crops (cv2). Source is RGB -> model channel c reads (2-c).
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < rw; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * kH + y) * imgW + x] =
            resized[(static_cast<size_t>(y) * rw + x) * 3 + (2 - c)] / 127.5f - 1.0f;

  std::array<int64_t, 4> ishape{1, 3, kH, imgW};
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {m.in_name.c_str()};
  const char* out_names[] = {m.out_name.c_str()};
  auto outs =
      const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  const float* logits = outs[0].GetTensorData<float>();
  auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int T = static_cast<int>(oshape[1]), C = static_cast<int>(oshape[2]);

  // CTC greedy decode: argmax per step, drop blank(0), collapse repeats.
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
      // softmax prob of argmax = exp(max - logsumexp)
      float mx = bestv, sum = 0.0f;
      for (int c = 0; c < C; ++c) sum += std::exp(row[c] - mx);
      res.score += 1.0f / sum;  // exp(max-max)/sum
      ++kept;
      if (best < (int)m.character.size()) text += m.character[best];
    }
    prev = best;
  }
  res.text = text;
  res.score = kept ? res.score / kept : 0.0f;
  return res;
}

}  // namespace mineru

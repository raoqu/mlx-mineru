// Copyright (c) mlx-mineru.
// Wired-table UNet — Stage 1: preprocess (keep-ratio bicubic resize to 1024 + normalize) +
// onnx inference -> line segmentation map. Faithful to MinerU TSRUnet.preprocess/infer.
#include "mineru/wired_table.hpp"

#include <algorithm>
#include <cmath>

#include "mineru/image_preprocess.hpp"  // resize_bicubic_rgb8
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {
constexpr int kInp = 1024;
// UNet normalization (ImageNet-style, RGB).
const std::array<float, 3> kMean{123.675f, 116.28f, 103.53f};
const std::array<float, 3> kStd{58.395f, 57.12f, 57.375f};
}  // namespace

struct WiredTableRecognizer::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-wired"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::string in_name, out_name;
};

WiredTableRecognizer::WiredTableRecognizer() : impl_(std::make_unique<Impl>()) {}
WiredTableRecognizer::~WiredTableRecognizer() = default;
WiredTableRecognizer::WiredTableRecognizer(WiredTableRecognizer&&) noexcept = default;
WiredTableRecognizer& WiredTableRecognizer::operator=(WiredTableRecognizer&&) noexcept = default;

WiredTableRecognizer WiredTableRecognizer::load(const std::string& onnx) {
  WiredTableRecognizer r;
  Impl& m = *r.impl_;
  m.session = std::make_unique<Ort::Session>(m.env, onnx.c_str(), m.opts);
  Ort::AllocatorWithDefaultOptions alloc;
  m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
  m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  return r;
}

std::vector<uint8_t> WiredTableRecognizer::segment(const std::vector<uint8_t>& rgb, int w, int h,
                                                   int& nh, int& nw) const {
  const Impl& m = *impl_;
  // imrescale: keep aspect, longest side -> 1024 (rounded).
  double scale = (double)kInp / std::max(w, h);
  nw = (int)(w * scale + 0.5);
  nh = (int)(h * scale + 0.5);
  std::vector<uint8_t> resized = resize_bicubic_rgb8(rgb, w, h, nw, nh);

  // NormalizeImage (RGB), CHW.
  std::vector<float> input(static_cast<size_t>(3) * nh * nw);
  for (int y = 0; y < nh; ++y)
    for (int x = 0; x < nw; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * nh + y) * nw + x] =
            (resized[(static_cast<size_t>(y) * nw + x) * 3 + c] - kMean[c]) / kStd[c];

  std::array<int64_t, 4> ishape{1, 3, nh, nw};
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {m.in_name.c_str()};
  const char* out_names[] = {m.out_name.c_str()};
  auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                        out_names, 1);
  // Output [1,1,nh,nw] int64 -> uint8 seg map.
  auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int oh = (int)oshape[oshape.size() - 2], ow = (int)oshape[oshape.size() - 1];
  const int64_t* seg = outs[0].GetTensorData<int64_t>();
  std::vector<uint8_t> out((size_t)oh * ow);
  for (size_t i = 0; i < out.size(); ++i) out[i] = (uint8_t)seg[i];
  nh = oh;
  nw = ow;
  return out;
}

}  // namespace mineru

// Copyright (c) mlx-mineru.
#include "mineru/table_cls.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "mineru/image_preprocess.hpp"  // resize_bilinear_rgb8
#include "mnn_runner.hpp"               // hybrid MNN path (faster; exact parity)
#include "onnxruntime_cxx_api.h"

namespace mineru {

// If a `<model>.mnn` sits next to the given `<model>.onnx`, return its path; else "".
static std::string sibling_mnn(const std::string& onnx_path) {
  if (onnx_path.size() > 5 && onnx_path.substr(onnx_path.size() - 5) == ".onnx") {
    std::string p = onnx_path.substr(0, onnx_path.size() - 5) + ".mnn";
    std::error_code ec;
    if (std::filesystem::exists(p, ec)) return p;
  }
  return {};
}

struct TableClassifier::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-tabcls"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;  // ORT path (fallback)
  std::unique_ptr<MnnRunner> mnn;         // MNN path (preferred when a .mnn sibling exists)
  std::string in_name, out_name;
};

TableClassifier::TableClassifier() : impl_(std::make_unique<Impl>()) {}
TableClassifier::~TableClassifier() = default;
TableClassifier::TableClassifier(TableClassifier&&) noexcept = default;
TableClassifier& TableClassifier::operator=(TableClassifier&&) noexcept = default;

TableClassifier TableClassifier::load(const std::string& onnx_path) {
  TableClassifier c;
  Impl& m = *c.impl_;
  if (std::string mp = sibling_mnn(onnx_path); !mp.empty())
    m.mnn = MnnRunner::load(mp, {"x"}, {"fetch_name_0"});
  if (!m.mnn) {  // no .mnn (or it failed to load) -> ONNX Runtime
    m.session = std::make_unique<Ort::Session>(m.env, onnx_path.c_str(), m.opts);
    Ort::AllocatorWithDefaultOptions alloc;
    m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
    m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  }
  return c;
}

TableClsResult TableClassifier::classify(const std::vector<uint8_t>& rgb, int w, int h) const {
  const Impl& m = *impl_;
  // resize shortest edge -> 256 (bilinear), center-crop 224, ImageNet-normalize, CHW.
  double scale = 256.0 / std::min(h, w);
  int rw = static_cast<int>(std::lround(w * scale)), rh = static_cast<int>(std::lround(h * scale));
  std::vector<uint8_t> r = resize_bilinear_rgb8(rgb, w, h, rw, rh);
  const int cw = 224, ch = 224;
  int x1 = std::max(0, (rw - cw) / 2), y1 = std::max(0, (rh - ch) / 2);
  if (rw < cw || rh < ch) throw std::runtime_error("table_cls: crop smaller than 224");

  static const float mean[3] = {0.485f, 0.456f, 0.406f};
  static const float std_[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> input(static_cast<size_t>(3) * ch * cw);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c) {
        uint8_t p = r[(static_cast<size_t>(y1 + y) * rw + (x1 + x)) * 3 + c];
        input[(static_cast<size_t>(c) * ch + y) * cw + x] = (p / 255.0f - mean[c]) / std_[c];
      }

  std::array<float, 2> prob;
  if (m.mnn) {
    std::vector<std::vector<float>> outs;
    std::vector<std::vector<int>> oshs;
    if (!m.mnn->run(input.data(), {1, 3, ch, cw}, outs, oshs) || outs.empty() || outs[0].size() < 2)
      throw std::runtime_error("table_cls: MNN inference failed");
    prob = {outs[0][0], outs[0][1]};
  } else {
    std::array<int64_t, 4> ishape{1, 3, ch, cw};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
    const char* in_names[] = {m.in_name.c_str()};
    const char* out_names[] = {m.out_name.c_str()};
    auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                          out_names, 1);
    const float* p = outs[0].GetTensorData<float>();
    prob = {p[0], p[1]};
  }

  TableClsResult res;
  res.probs = {prob[0], prob[1]};
  res.cls_id = (prob[1] > prob[0]) ? 1 : 0;
  res.score = prob[res.cls_id];
  res.label = res.cls_id == 0 ? "wired_table" : "wireless_table";
  return res;
}

}  // namespace mineru

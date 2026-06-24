// Copyright (c) mlx-mineru.
#include "mineru/layout_det.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "mineru/image_preprocess.hpp"  // resize_bicubic_rgb8
#include "nlohmann/json.hpp"
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {
constexpr int kSize = 800;
std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("layout: cannot open " + p);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
}  // namespace

struct LayoutDetector::Impl {
  float conf;
  std::vector<std::string> id2label;  // index = class id
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-layout"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
};

LayoutDetector::LayoutDetector() : impl_(std::make_unique<Impl>()) {}
LayoutDetector::~LayoutDetector() = default;
LayoutDetector::LayoutDetector(LayoutDetector&&) noexcept = default;
LayoutDetector& LayoutDetector::operator=(LayoutDetector&&) noexcept = default;

LayoutDetector LayoutDetector::load(const std::string& model_dir, float conf) {
  LayoutDetector d;
  Impl& m = *d.impl_;
  m.conf = conf;
  nlohmann::json cfg = nlohmann::json::parse(read_file(model_dir + "/config.json"));
  int n = 0;
  for (auto& [k, v] : cfg["id2label"].items()) n = std::max(n, std::stoi(k) + 1);
  m.id2label.assign(n, "");
  for (auto& [k, v] : cfg["id2label"].items()) m.id2label[std::stoi(k)] = v.get<std::string>();
  m.opts.SetIntraOpNumThreads(0);  // ORT default
  m.session = std::make_unique<Ort::Session>(m.env, (model_dir + "/layout.onnx").c_str(), m.opts);
  return d;
}

std::vector<LayoutBox> LayoutDetector::detect_800(const std::vector<uint8_t>& rgb, int target_w,
                                                  int target_h) const {
  const Impl& m = *impl_;
  // /255, NCHW.
  std::vector<float> input(static_cast<size_t>(3) * kSize * kSize);
  for (int y = 0; y < kSize; ++y)
    for (int x = 0; x < kSize; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * kSize + y) * kSize + x] =
            rgb[(static_cast<size_t>(y) * kSize + x) * 3 + c] / 255.0f;

  std::array<int64_t, 4> ishape{1, 3, kSize, kSize};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {"pixel_values"};
  const char* out_names[] = {"logits", "pred_boxes", "order_logits"};
  auto outs =
      const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 3);

  const float* logits = outs[0].GetTensorData<float>();      // (1,300,ncl)
  const float* boxes = outs[1].GetTensorData<float>();       // (1,300,4) cx,cy,w,h
  const float* order_logits = outs[2].GetTensorData<float>();  // (1,Q,Q)
  auto lshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int nq = static_cast<int>(lshape[1]), ncl = static_cast<int>(lshape[2]);

  // Reading-order head decode (_get_order_seqs): per-query vote = sum_{i<q} sig(ol[i,q]) +
  // sum_{a>q} (1 - sig(ol[q,a])); argsort(votes) -> pointers; seq[pointer]=rank.
  auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
  std::vector<double> votes(nq, 0.0);
  for (int q = 0; q < nq; ++q) {
    double v = 0;
    for (int i = 0; i < q; ++i) v += sig(order_logits[(size_t)i * nq + q]);
    for (int a = q + 1; a < nq; ++a) v += 1.0 - sig(order_logits[(size_t)q * nq + a]);
    votes[q] = v;
  }
  std::vector<int> ptr(nq);
  std::iota(ptr.begin(), ptr.end(), 0);
  std::stable_sort(ptr.begin(), ptr.end(), [&](int a, int b) { return votes[a] < votes[b]; });
  std::vector<int> order_seq(nq);
  for (int r = 0; r < nq; ++r) order_seq[ptr[r]] = r;

  // Box decode (cx,cy,w,h -> corners) scaled to target.
  std::vector<std::array<float, 4>> corners(nq);
  for (int q = 0; q < nq; ++q) {
    float cx = boxes[q * 4 + 0], cy = boxes[q * 4 + 1], w = boxes[q * 4 + 2], h = boxes[q * 4 + 3];
    corners[q] = {(cx - 0.5f * w) * target_w, (cy - 0.5f * h) * target_h,
                  (cx + 0.5f * w) * target_w, (cy + 0.5f * h) * target_h};
  }
  // sigmoid scores over all query x class, topk = nq descending (stable on ties).
  std::vector<float> scores(static_cast<size_t>(nq) * ncl);
  for (size_t i = 0; i < scores.size(); ++i) scores[i] = 1.0f / (1.0f + std::exp(-logits[i]));
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return scores[a] > scores[b]; });

  auto clip = [](int v, int hi) { return std::max(0, std::min(hi, v)); };
  std::vector<std::pair<int, LayoutBox>> kept;  // (order_seq, box)
  for (int k = 0; k < nq; ++k) {
    int fi = order[k];
    float s = scores[fi];
    if (s < m.conf) continue;
    int cls = fi % ncl, q = fi / ncl;
    const auto& c = corners[q];
    LayoutBox b;
    b.cls_id = cls;
    b.label = (cls < (int)m.id2label.size()) ? m.id2label[cls] : std::to_string(cls);
    b.score = s;
    b.bbox = {clip((int)std::floor(c[0]), target_w), clip((int)std::floor(c[1]), target_h),
              clip((int)std::ceil(c[2]), target_w), clip((int)std::ceil(c[3]), target_h)};
    kept.emplace_back(order_seq[q], std::move(b));
  }
  // Sort by reading-order, assign 1-based index.
  std::stable_sort(kept.begin(), kept.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<LayoutBox> res;
  res.reserve(kept.size());
  for (int i = 0; i < (int)kept.size(); ++i) {
    kept[i].second.index = i + 1;
    res.push_back(std::move(kept[i].second));
  }
  return res;
}

std::vector<LayoutBox> LayoutDetector::detect(const std::vector<uint8_t>& rgb, int w, int h) const {
  std::vector<uint8_t> r = resize_bicubic_rgb8(rgb, w, h, kSize, kSize);
  return detect_800(r, w, h);  // boxes scaled back to page (w,h)
}

}  // namespace mineru

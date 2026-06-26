// Copyright (c) mlx-mineru.
#include "mineru/layout_det.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "mineru/image_preprocess.hpp"  // resize_bicubic_rgb8
#include "mnn_runner.hpp"               // MNN/Metal backbone (split path)
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
  std::unique_ptr<Ort::Session> session;        // full model (fallback)
  std::unique_ptr<MnnRunner> backbone;          // split: conv backbone+encoder on MNN/Metal
  std::unique_ptr<Ort::Session> decoder;        // split: RT-DETR decoder on ORT
  bool split = false;
};

LayoutDetector::LayoutDetector() : impl_(std::make_unique<Impl>()) {}
LayoutDetector::~LayoutDetector() = default;
LayoutDetector::LayoutDetector(LayoutDetector&&) noexcept = default;
LayoutDetector& LayoutDetector::operator=(LayoutDetector&&) noexcept = default;

LayoutDetector LayoutDetector::load(const std::string& model_dir, float conf) {
  LayoutDetector d;
  Impl& m = *d.impl_;
  m.conf = conf;
  // MinerU overrides config.json id2label with its hardcoded PP_DOCLAYOUT_V2_LABELS
  // (config.json collapses display_formula/inline_formula/reference into "formula"/etc.).
  // Use MinerU's authoritative names so downstream label-based routing matches.
  m.id2label = {"abstract",      "algorithm",   "aside_text",     "chart",
                "content",       "display_formula", "doc_title",  "figure_title",
                "footer",        "footer_image", "footnote",      "formula_number",
                "header",        "header_image", "image",         "inline_formula",
                "number",        "paragraph_title", "reference",  "reference_content",
                "seal",          "table",        "text",          "vertical_text",
                "vision_footnote"};
  m.opts.SetIntraOpNumThreads(0);  // ORT default
  // Split path: the conv backbone+encoder (61% of the time, Metal-clean) runs on MNN/Metal and the
  // Metal-hostile RT-DETR decoder (TopK/ScatterND/GridSample deformable attention) on ORT. The
  // single handoff is the encoder memory [1,13125,256]. scripts/split_layout_onnx.py builds both
  // (constants baked into the decoder; valid only for the fixed 800x800 input). Falls back to the
  // full layout.onnx when the split files are absent or the MNN backbone won't load.
  namespace fsx = std::filesystem;
  std::error_code ec;
  std::string bb = model_dir + "/layout_backbone.mnn", dec = model_dir + "/layout_decoder.onnx";
  if (fsx::exists(bb, ec) && fsx::exists(dec, ec)) {
    m.backbone = MnnRunner::load(bb, {"pixel_values"}, {"/m/model/Concat_3_output_0"});
    if (m.backbone) {
      m.decoder = std::make_unique<Ort::Session>(m.env, dec.c_str(), m.opts);
      m.split = true;
      // Warm up: compile the Metal backbone shaders once at load so the first page isn't penalized.
      std::vector<float> dummy(static_cast<size_t>(3) * kSize * kSize, 0.0f);
      std::vector<std::vector<float>> o;
      std::vector<std::vector<int>> s;
      m.backbone->run(dummy.data(), {1, 3, kSize, kSize}, o, s);
    }
  }
  if (!m.split)
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

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* in_names[] = {"pixel_values"};
  const char* out_names[] = {"logits", "pred_boxes", "order_logits"};
  std::vector<Ort::Value> outs;
  std::vector<float> mem_buf;  // backbone memory (split path; kept alive for the decoder Run)
  if (m.split) {
    // MNN/Metal backbone -> encoder memory [1,13125,256] -> ORT decoder.
    std::vector<std::vector<float>> mo;
    std::vector<std::vector<int>> ms;
    if (!const_cast<MnnRunner&>(*m.backbone).run(input.data(), {1, 3, kSize, kSize}, mo, ms) ||
        mo.empty() || ms[0].size() < 3)
      throw std::runtime_error("layout: MNN backbone failed");
    mem_buf = std::move(mo[0]);
    std::array<int64_t, 3> mshape{ms[0][0], ms[0][1], ms[0][2]};
    Ort::Value memv =
        Ort::Value::CreateTensor<float>(mem, mem_buf.data(), mem_buf.size(), mshape.data(), 3);
    const char* dec_in[] = {"/m/model/Concat_3_output_0"};
    outs = const_cast<Ort::Session&>(*m.decoder).Run(Ort::RunOptions{nullptr}, dec_in, &memv, 1,
                                                     out_names, 3);
  } else {
    std::array<int64_t, 4> ishape{1, 3, kSize, kSize};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), ishape.data(), 4);
    outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                     out_names, 3);
  }

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

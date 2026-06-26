// Copyright (c) mlx-mineru.
#include "mnn_runner.hpp"

#include <cstdlib>
#include <cstring>

#ifdef MINERU_HAVE_MNN
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#endif

namespace mineru {

#ifdef MINERU_HAVE_MNN
using namespace MNN::Express;

namespace {
// Backend selected by MINERU_MNN_BACKEND (cpu|metal|coreml|auto), default metal.
//   metal  — MNN_FORWARD_METAL (default), GPU; large speedups on the heavier CV models
//            (unet/ocr_rec/ocr_det) on Apple Silicon at fp32 (Precision_High) with golden-
//            verified parity. Big win across multi-hundred-page PDFs where CV cost compounds.
//   cpu    — MNN_FORWARD_CPU, fp32, the verified-parity baseline.
//   coreml — MNN_FORWARD_NN (ANE/GPU); MNN's CoreML codegen currently fails to compile these
//            models (spec-version / unsupported-op), so it falls back to CPU in practice.
//   auto   — same as metal (try Metal, fall back to CPU).
// Any non-CPU backend that fails to load OR fails at first inference transparently rebuilds the
// module on CPU, so a backend op gap degrades gracefully instead of breaking the pipeline.
MNNForwardType requested_backend() {
  const char* e = std::getenv("MINERU_MNN_BACKEND");
  if (!e || !*e) return MNN_FORWARD_METAL;  // default: GPU
  std::string s(e);
  if (s == "cpu") return MNN_FORWARD_CPU;
  if (s == "coreml" || s == "nn") return MNN_FORWARD_NN;
  return MNN_FORWARD_METAL;  // "metal", "auto", anything else
}

std::shared_ptr<Module> load_on(MNNForwardType fwd, const std::string& mnn_path,
                                const std::vector<std::string>& in,
                                const std::vector<std::string>& out, bool fp16,
                                std::shared_ptr<Executor::RuntimeManager>& rtmgr_out) {
  MNN::ScheduleConfig sc;
  sc.type = fwd;
  sc.numThread = 4;
  MNN::BackendConfig bc;
  // Default fp32 (Precision_High): near-exact parity with CPU/ONNX (golden-passing) and required
  // by the strict table_cls prob tolerance. fp16 (Precision_Normal) is opt-in per model (caller's
  // `fp16`): ~15-25% faster on Metal, safe for argmax/CTC outputs (ocr_det/ocr_rec, golden-verified).
  bc.precision = fp16 ? MNN::BackendConfig::Precision_Normal : MNN::BackendConfig::Precision_High;
  bc.power = MNN::BackendConfig::Power_High;
  sc.backendConfig = &bc;
  std::shared_ptr<Executor::RuntimeManager> rtmgr(
      Executor::RuntimeManager::createRuntimeManager(sc));
  if (!rtmgr) return nullptr;
  Module::Config mc;
  mc.shapeMutable = true;  // allow dynamic input H/W (ocr_rec width, unet, slanet)
  mc.rearrange = false;
  std::shared_ptr<Module> net(Module::load(in, out, mnn_path.c_str(), rtmgr, &mc));
  if (!net) return nullptr;
  rtmgr_out = rtmgr;
  return net;
}

// Copy MNN outputs into flattened float buffers (int label outputs normalized to float).
bool read_outputs(const std::vector<VARP>& result, std::vector<std::vector<float>>& outs,
                  std::vector<std::vector<int>>& out_shapes) {
  if (result.empty()) return false;
  outs.assign(result.size(), {});
  out_shapes.assign(result.size(), {});
  for (size_t i = 0; i < result.size(); ++i) {
    VARP v = result[i];
    if (!v.get()) return false;
    auto info = v->getInfo();
    if (!info) return false;
    out_shapes[i].assign(info->dim.begin(), info->dim.end());
    const int n = info->size;
    outs[i].resize(n);
    if (info->type.code == halide_type_float) {
      const float* p = v->readMap<float>();
      if (!p) return false;
      std::memcpy(outs[i].data(), p, n * sizeof(float));
    } else if (info->type.code == halide_type_int && info->type.bits == 32) {
      const int32_t* p = v->readMap<int32_t>();
      if (!p) return false;
      for (int k = 0; k < n; ++k) outs[i][k] = static_cast<float>(p[k]);
    } else if (info->type.code == halide_type_int && info->type.bits == 64) {
      const int64_t* p = v->readMap<int64_t>();
      if (!p) return false;
      for (int k = 0; k < n; ++k) outs[i][k] = static_cast<float>(p[k]);
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

struct MnnRunner::Impl {
  std::shared_ptr<Executor::RuntimeManager> rtmgr;
  std::shared_ptr<Module> net;
  std::vector<std::string> in_names;
  std::vector<std::string> out_names;
  std::string path;
  MNNForwardType backend = MNN_FORWARD_CPU;
  bool fp16 = false;       // Precision_Normal (half) requested at load
  bool fell_back = false;  // already rebuilt on CPU after a non-CPU failure?
};

MnnRunner::MnnRunner() : impl_(std::make_unique<Impl>()) {}
MnnRunner::~MnnRunner() = default;

std::unique_ptr<MnnRunner> MnnRunner::load(const std::string& mnn_path,
                                           const std::vector<std::string>& input_names,
                                           const std::vector<std::string>& output_names,
                                           bool fp16) {
  MNNForwardType fwd = requested_backend();
  std::shared_ptr<Executor::RuntimeManager> rtmgr;
  std::shared_ptr<Module> net = load_on(fwd, mnn_path, input_names, output_names, fp16, rtmgr);
  if (!net && fwd != MNN_FORWARD_CPU) {  // backend unavailable/load failed -> CPU fallback
    fwd = MNN_FORWARD_CPU;
    net = load_on(fwd, mnn_path, input_names, output_names, fp16, rtmgr);
  }
  if (!net) return nullptr;
  std::unique_ptr<MnnRunner> r(new MnnRunner());
  r->impl_->rtmgr = rtmgr;
  r->impl_->net = net;
  r->impl_->in_names = input_names;
  r->impl_->out_names = output_names;
  r->impl_->path = mnn_path;
  r->impl_->backend = fwd;
  r->impl_->fp16 = fp16;
  r->impl_->fell_back = (fwd == MNN_FORWARD_CPU);
  return r;
}

bool MnnRunner::run(const float* input, const std::array<int, 4>& nchw,
                    std::vector<std::vector<float>>& outs, std::vector<std::vector<int>>& out_shapes) {
  VARP x = _Input({nchw[0], nchw[1], nchw[2], nchw[3]}, NCHW, halide_type_of<float>());
  std::memcpy(x->writeMap<float>(), input,
              static_cast<size_t>(nchw[0]) * nchw[1] * nchw[2] * nchw[3] * sizeof(float));
  std::vector<VARP> result = impl_->net->onForward({x});
  if ((result.empty() || !result[0].get()) && !impl_->fell_back &&
      impl_->backend != MNN_FORWARD_CPU) {
    // The chosen GPU/ANE backend could not execute this model (e.g. CoreML op gap surfaces only
    // at first forward) — rebuild on CPU once and retry, so the model still runs.
    std::shared_ptr<Executor::RuntimeManager> rtmgr;
    std::shared_ptr<Module> net = load_on(MNN_FORWARD_CPU, impl_->path, impl_->in_names,
                                          impl_->out_names, impl_->fp16, rtmgr);
    if (net) {
      impl_->rtmgr = rtmgr;
      impl_->net = net;
      impl_->backend = MNN_FORWARD_CPU;
      impl_->fell_back = true;
      VARP x2 = _Input({nchw[0], nchw[1], nchw[2], nchw[3]}, NCHW, halide_type_of<float>());
      std::memcpy(x2->writeMap<float>(), input,
                  static_cast<size_t>(nchw[0]) * nchw[1] * nchw[2] * nchw[3] * sizeof(float));
      result = impl_->net->onForward({x2});
    }
  }
  return read_outputs(result, outs, out_shapes);
}

#else  // !MINERU_HAVE_MNN — stub so the pipeline builds without MNN (always falls back to ORT).

struct MnnRunner::Impl {};
MnnRunner::MnnRunner() = default;
MnnRunner::~MnnRunner() = default;
std::unique_ptr<MnnRunner> MnnRunner::load(const std::string&, const std::vector<std::string>&,
                                           const std::vector<std::string>&) {
  return nullptr;
}
bool MnnRunner::run(const float*, const std::array<int, 4>&, std::vector<std::vector<float>>&,
                    std::vector<std::vector<int>>&) {
  return false;
}

#endif

}  // namespace mineru

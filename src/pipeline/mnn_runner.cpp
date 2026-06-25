// Copyright (c) mlx-mineru.
#include "mnn_runner.hpp"

#include <cstring>

#ifdef MINERU_HAVE_MNN
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#endif

namespace mineru {

#ifdef MINERU_HAVE_MNN
using namespace MNN::Express;

struct MnnRunner::Impl {
  std::shared_ptr<Executor::RuntimeManager> rtmgr;
  std::shared_ptr<Module> net;
  std::vector<std::string> in_names;
};

MnnRunner::MnnRunner() : impl_(std::make_unique<Impl>()) {}
MnnRunner::~MnnRunner() = default;

std::unique_ptr<MnnRunner> MnnRunner::load(const std::string& mnn_path,
                                           const std::vector<std::string>& input_names,
                                           const std::vector<std::string>& output_names) {
  MNN::ScheduleConfig sc;
  sc.type = MNN_FORWARD_CPU;  // CPU = verified-parity win; Metal/CoreML have op gaps.
  sc.numThread = 4;
  std::shared_ptr<Executor::RuntimeManager> rtmgr(
      Executor::RuntimeManager::createRuntimeManager(sc));
  if (!rtmgr) return nullptr;
  Module::Config mc;
  mc.shapeMutable = true;  // allow dynamic input H/W (ocr_rec width, unet, slanet)
  mc.rearrange = false;
  std::shared_ptr<Module> net(
      Module::load(input_names, output_names, mnn_path.c_str(), rtmgr, &mc));
  if (!net) return nullptr;
  std::unique_ptr<MnnRunner> r(new MnnRunner());
  r->impl_->rtmgr = rtmgr;
  r->impl_->net = net;
  r->impl_->in_names = input_names;
  return r;
}

bool MnnRunner::run(const float* input, const std::array<int, 4>& nchw,
                    std::vector<std::vector<float>>& outs, std::vector<std::vector<int>>& out_shapes) {
  VARP x = _Input({nchw[0], nchw[1], nchw[2], nchw[3]}, NCHW, halide_type_of<float>());
  std::memcpy(x->writeMap<float>(), input,
              static_cast<size_t>(nchw[0]) * nchw[1] * nchw[2] * nchw[3] * sizeof(float));
  std::vector<VARP> result = impl_->net->onForward({x});
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
    // Normalize int label outputs (e.g. UNet) to float; float outputs pass through.
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

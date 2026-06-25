// Copyright (c) mlx-mineru.
#include "mnn_runner.hpp"

#include <cstring>

#ifdef MINERU_HAVE_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#endif

namespace mineru {

#ifdef MINERU_HAVE_MNN

struct MnnRunner::Impl {
  std::shared_ptr<MNN::Interpreter> net;
  MNN::Session* session = nullptr;
};

MnnRunner::MnnRunner() : impl_(std::make_unique<Impl>()) {}

MnnRunner::~MnnRunner() {
  if (impl_ && impl_->net && impl_->session) impl_->net->releaseSession(impl_->session);
}

std::unique_ptr<MnnRunner> MnnRunner::load(const std::string& mnn_path) {
  std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(mnn_path.c_str()),
                                        [](MNN::Interpreter* p) { MNN::Interpreter::destroy(p); });
  if (!net) return nullptr;
  MNN::ScheduleConfig cfg;
  cfg.type = MNN_FORWARD_CPU;  // CPU is the exact-parity win; the Metal backend had op gaps.
  cfg.numThread = 4;
  MNN::BackendConfig bc;
  bc.precision = MNN::BackendConfig::Precision_High;  // fp32 — keep parity with ORT
  cfg.backendConfig = &bc;
  MNN::Session* s = net->createSession(cfg);
  if (!s) return nullptr;
  std::unique_ptr<MnnRunner> r(new MnnRunner());
  r->impl_->net = net;
  r->impl_->session = s;
  return r;
}

std::vector<float> MnnRunner::run(const float* input, const std::array<int, 4>& nchw,
                                  std::vector<int>& out_shape) {
  auto& net = impl_->net;
  MNN::Session* s = impl_->session;
  MNN::Tensor* in = net->getSessionInput(s, nullptr);
  net->resizeTensor(in, {nchw[0], nchw[1], nchw[2], nchw[3]});
  net->resizeSession(s);

  std::unique_ptr<MNN::Tensor> host(new MNN::Tensor(in, MNN::Tensor::CAFFE));  // CAFFE = NCHW
  std::memcpy(host->host<float>(), input,
              static_cast<size_t>(nchw[0]) * nchw[1] * nchw[2] * nchw[3] * sizeof(float));
  in->copyFromHostTensor(host.get());

  if (net->runSession(s) != MNN::NO_ERROR) return {};

  MNN::Tensor* out = net->getSessionOutput(s, nullptr);
  if (!out) return {};
  std::unique_ptr<MNN::Tensor> oh(new MNN::Tensor(out, MNN::Tensor::CAFFE));
  out->copyToHostTensor(oh.get());
  out_shape = oh->shape();
  const int n = oh->elementSize();
  std::vector<float> res(n);
  // Normalize any numeric output dtype to float (e.g. UNet emits int32/int64 segmentation
  // labels; classifiers emit float logits) so callers get a single representation.
  halide_type_t ty = oh->getType();
  if (ty.code == halide_type_float) {
    const float* p = oh->host<float>();
    for (int i = 0; i < n; ++i) res[i] = p[i];
  } else if (ty.code == halide_type_int && ty.bits == 32) {
    const int32_t* p = oh->host<int32_t>();
    for (int i = 0; i < n; ++i) res[i] = static_cast<float>(p[i]);
  } else if (ty.code == halide_type_int && ty.bits == 64) {
    const int64_t* p = oh->host<int64_t>();
    for (int i = 0; i < n; ++i) res[i] = static_cast<float>(p[i]);
  } else if (ty.code == halide_type_uint && ty.bits == 8) {
    const uint8_t* p = oh->host<uint8_t>();
    for (int i = 0; i < n; ++i) res[i] = static_cast<float>(p[i]);
  } else {
    return {};  // unsupported dtype
  }
  return res;
}

#else  // !MINERU_HAVE_MNN — stub so the pipeline builds without MNN (always falls back to ORT).

struct MnnRunner::Impl {};
MnnRunner::MnnRunner() = default;
MnnRunner::~MnnRunner() = default;
std::unique_ptr<MnnRunner> MnnRunner::load(const std::string&) { return nullptr; }
std::vector<float> MnnRunner::run(const float*, const std::array<int, 4>&, std::vector<int>&) {
  return {};
}

#endif

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Minimal single-model MNN runner for the *hybrid* pipeline path. Only the two models MNN
// runs faster with exact parity (table-cls PP-LCNet, wired-table UNet) opt into this; every
// other model stays on ONNX Runtime. load() returns nullptr when MNN is unavailable or the
// model can't be loaded, so callers transparently fall back to ORT.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

class MnnRunner {
 public:
  // Load a `.mnn` model (CPU backend, fp32). Returns nullptr on any failure.
  static std::unique_ptr<MnnRunner> load(const std::string& mnn_path);
  ~MnnRunner();

  // Run one NCHW float input through the model; returns the first output flattened row-major
  // (NCHW), filling out_shape. Empty vector on failure.
  std::vector<float> run(const float* input, const std::array<int, 4>& nchw,
                         std::vector<int>& out_shape);

 private:
  MnnRunner();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

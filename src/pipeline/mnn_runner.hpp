// Copyright (c) mlx-mineru.
// Minimal MNN runner for the *hybrid* pipeline path, on MNN's Module/Express API (supports
// control flow `If`, dynamic shapes, and multi-output models — the legacy Interpreter/Session
// API does not). Used by the pipeline models MNN runs faster than ONNX Runtime with verified
// parity. load() returns nullptr when MNN is unavailable or the model can't be loaded, so
// callers transparently fall back to ONNX Runtime.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

class MnnRunner {
 public:
  // Load a `.mnn` model with the given input/output tensor names (Module API needs them).
  // Returns nullptr on any failure.
  static std::unique_ptr<MnnRunner> load(const std::string& mnn_path,
                                         const std::vector<std::string>& input_names,
                                         const std::vector<std::string>& output_names);
  ~MnnRunner();

  // Run a single NCHW float input; fills `outs` (one flattened row-major buffer per requested
  // output, in the load() order; int label outputs are converted to float) and `out_shapes`.
  // Returns false on failure.
  bool run(const float* input, const std::array<int, 4>& nchw,
           std::vector<std::vector<float>>& outs, std::vector<std::vector<int>>& out_shapes);

 private:
  MnnRunner();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

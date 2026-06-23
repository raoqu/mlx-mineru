// Copyright (c) mlx-mineru.
// Pipeline P1 foundation: ONNX Runtime C++ loads and runs the exported layout
// model. Verifies the integration end-to-end (session, tensor I/O, output shapes).
#include <array>
#include <iostream>
#include <thread>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "test_util.hpp"

int main(int argc, char** argv) {
  std::string onnx = (argc > 1) ? argv[1] : "models/pipeline/Layout/layout.onnx";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mlx-mineru");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
  Ort::Session session(env, onnx.c_str(), opts);

  CHECK_MSG(session.GetInputCount() == 1, "expected 1 input");
  CHECK_MSG(session.GetOutputCount() == 3, "expected 3 outputs (logits/pred_boxes/order_logits)");

  // pixel_values (1,3,800,800), zeros — just exercising the run path.
  std::vector<float> input(static_cast<size_t>(1) * 3 * 800 * 800, 0.0f);
  std::array<int64_t, 4> ishape{1, 3, 800, 800};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), ishape.data(), 4);

  const char* in_names[] = {"pixel_values"};
  const char* out_names[] = {"logits", "pred_boxes", "order_logits"};
  auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 3);

  CHECK(outs.size() == 3);
  // logits (1,300,25), pred_boxes (1,300,4), order_logits (1,300,300)
  std::array<std::array<int64_t, 3>, 3> want = {{{1, 300, 25}, {1, 300, 4}, {1, 300, 300}}};
  for (int i = 0; i < 3; ++i) {
    auto shape = outs[i].GetTensorTypeAndShapeInfo().GetShape();
    bool ok = shape.size() == 3 && shape[0] == want[i][0] && shape[1] == want[i][1] &&
              shape[2] == want[i][2];
    CHECK_MSG(ok, std::string(out_names[i]) + ": unexpected shape");
    if (!ok) {
      std::cerr << "  got [";
      for (auto d : shape) std::cerr << d << " ";
      std::cerr << "]\n";
    }
  }
  std::cerr << "onnx_smoke: layout.onnx loaded and ran; output shapes correct\n";
  return TEST_SUMMARY();
}

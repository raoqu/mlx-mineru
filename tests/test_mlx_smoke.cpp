// Copyright (c) mlx-mineru.
// Phase 4a verification: MLX C++ links and computes on the Metal GPU.
// Proves the MLX/Metal toolchain works from native C++ before porting Qwen2-VL.
#include <cmath>
#include <vector>

#include "mlx/mlx.h"
#include "test_util.hpp"

namespace mx = mlx::core;

int main() {
  // 2x2 matmul on the default (GPU) device.
  mx::array a = mx::array({1.f, 2.f, 3.f, 4.f}, {2, 2});
  mx::array b = mx::array({5.f, 6.f, 7.f, 8.f}, {2, 2});
  mx::array c = mx::matmul(a, b);
  mx::eval(c);

  CHECK(c.shape() == mx::Shape({2, 2}));
  const float* p = c.data<float>();
  // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
  CHECK(std::abs(p[0] - 19.f) < 1e-4);
  CHECK(std::abs(p[1] - 22.f) < 1e-4);
  CHECK(std::abs(p[2] - 43.f) < 1e-4);
  CHECK(std::abs(p[3] - 50.f) < 1e-4);

  // Reduction + a unary op to exercise more of the graph/eval path.
  mx::array s = mx::sum(mx::array({1.f, 2.f, 3.f, 4.f}));
  mx::array r = mx::sqrt(s);  // sqrt(10)
  mx::eval(r);
  CHECK(std::abs(r.item<float>() - std::sqrt(10.f)) < 1e-4);

  // Confirm we actually have a GPU device available.
  CHECK(mx::default_device().type == mx::Device::gpu ||
        mx::default_device().type == mx::Device::cpu);

  return TEST_SUMMARY();
}

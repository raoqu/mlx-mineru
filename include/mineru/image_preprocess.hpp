// Copyright (c) mlx-mineru.
// Qwen2-VL image preprocessing — faithful port of transformers
// Qwen2VLImageProcessor (smart_resize -> PIL bicubic resize -> rescale ->
// normalize -> patchify). Produces the flat `pixel_values` + grid (t,h,w) the
// vision encoder consumes. This is what MinerU's MLX VLM path feeds the model.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mineru {

struct VisionInput {
  std::vector<float> pixel_values;          // [grid_t*grid_h*grid_w, 1176] row-major
  std::array<int, 3> grid_thw{0, 0, 0};     // (grid_t, grid_h, grid_w)
  int seq_len() const { return grid_thw[0] * grid_thw[1] * grid_thw[2]; }
  int feat_dim() const { return 1176; }     // channel*temporal*patch*patch = 3*2*14*14
};

struct Qwen2VLImageConfig {
  int patch_size = 14;
  int temporal_patch_size = 2;
  int merge_size = 2;
  int min_pixels = 50176;     // 56*56*16
  int max_pixels = 1605632;   // from preprocessor_config.json
  std::array<float, 3> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
  std::array<float, 3> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
};

// (h_bar, w_bar) per Qwen2VL smart_resize.
std::array<int, 2> smart_resize(int height, int width, int factor, int min_pixels, int max_pixels);

// rgb: width*height*3 uint8 (top-down rows). Returns pixel_values + grid.
VisionInput preprocess_image(const std::vector<uint8_t>& rgb, int width, int height,
                             const Qwen2VLImageConfig& cfg = {});

// PIL-faithful 8-bit bicubic resize (exposed for testing). in/out RGB8.
std::vector<uint8_t> resize_bicubic_rgb8(const std::vector<uint8_t>& rgb, int in_w, int in_h,
                                         int out_w, int out_h);

// Bilinear resize, half-pixel-center mapping (cv2 INTER_LINEAR convention; float
// path, not cv2's fixed-point — used by the pipeline OCR/table preprocessing).
std::vector<uint8_t> resize_bilinear_rgb8(const std::vector<uint8_t>& rgb, int in_w, int in_h,
                                          int out_w, int out_h);

}  // namespace mineru

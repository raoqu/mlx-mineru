// Copyright (c) mlx-mineru.
// RGB8 resize through real OpenCV (cv2 4.13.0) when available, so pipeline preprocess
// matches MinerU's cv2.resize byte-for-byte. Falls back to the float bilinear/bicubic
// (image_preprocess) when OpenCV isn't linked.
#pragma once

#include <cstdint>
#include <vector>

namespace mineru {

// cv2 interpolation codes (match cv::INTER_*).
enum CvInterp { kInterNearest = 0, kInterLinear = 1, kInterCubic = 2, kInterArea = 3 };

// Resize interleaved RGB8 (in_w*in_h*3) to (out_w*out_h*3).
std::vector<uint8_t> resize_rgb8_cv(const std::vector<uint8_t>& rgb, int in_w, int in_h,
                                    int out_w, int out_h, int interp);

// True when the build links OpenCV (so resize_rgb8_cv is byte-exact vs cv2.resize).
bool have_opencv_resize();

}  // namespace mineru

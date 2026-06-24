// Copyright (c) mlx-mineru.
// OCR text detection (PP-OCRv6 DBNet, pipeline backend): page image -> text-line
// quad boxes. Faithful port of MinerU: DetResizeForTest (max-960 /32) + Normalize +
// ocr_det.onnx + DBPostProcess (box_type="quad").
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

struct DetBox {
  std::array<std::array<int, 2>, 4> pts;  // 4 corners (x,y), ordered like MinerU
  float score;
};

// DB post-process on a probability map (Hd x Wd, row-major float in [0,1]).
// shape = {src_h, src_w, ratio_h, ratio_w}; boxes are scaled to src_w x src_h.
std::vector<DetBox> db_postprocess(const std::vector<float>& probmap, int Hd, int Wd,
                                   const std::array<double, 4>& shape, float thresh = 0.3f,
                                   float box_thresh = 0.6f, float unclip_ratio = 1.5f);

// End-to-end detector: RGB page image -> text-line boxes (in source pixel coords).
class TextDetector {
 public:
  TextDetector();
  ~TextDetector();
  TextDetector(TextDetector&&) noexcept;
  TextDetector& operator=(TextDetector&&) noexcept;

  static TextDetector load(const std::string& onnx_path);

  // rgb: interleaved RGB8, w*h*3 bytes. Returns boxes scaled to (w,h).
  std::vector<DetBox> detect(const std::vector<uint8_t>& rgb, int w, int h) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

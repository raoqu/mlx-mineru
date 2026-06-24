// Copyright (c) mlx-mineru.
// Full OCR for a page (pipeline backend), faithful to MinerU PytorchPaddleOCR:
// det -> sorted_boxes -> merge_det_boxes -> rotate-crop -> batched rec -> drop_score.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mineru/ocr_det.hpp"
#include "mineru/ocr_rec.hpp"

namespace mineru {

struct OcrLine {
  std::array<std::array<float, 2>, 4> box;  // quad in source pixel coords
  std::string text;
  float score;
};

// --- exposed for unit tests / reuse (faithful ports of mineru/utils/ocr_utils.py) ---
using Quad = std::array<std::array<float, 2>, 4>;
std::vector<Quad> sorted_boxes(std::vector<Quad> boxes);
std::vector<Quad> merge_det_boxes(const std::vector<Quad>& boxes);
// Crop + (optional) 90-deg rotate for tall crops, like get_rotate_crop_image_for_text_rec.
std::vector<uint8_t> get_rotate_crop_image(const std::vector<uint8_t>& rgb, int w, int h,
                                           const Quad& box, int& out_w, int& out_h);

class OcrPipeline {
 public:
  static OcrPipeline load(const std::string& det_onnx, const std::string& rec_onnx,
                          const std::string& dict);

  // rgb: w*h*3 RGB8 page image. drop_score default 0.5 (MinerU text path).
  std::vector<OcrLine> run(const std::vector<uint8_t>& rgb, int w, int h,
                           float drop_score = 0.5f) const;

  const TextDetector& detector() const { return det_; }
  const TextRecognizer& recognizer() const { return rec_; }

 private:
  OcrPipeline(TextDetector d, TextRecognizer r) : det_(std::move(d)), rec_(std::move(r)) {}
  TextDetector det_;
  TextRecognizer rec_;
};

}  // namespace mineru

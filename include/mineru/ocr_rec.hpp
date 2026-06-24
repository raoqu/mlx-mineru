// Copyright (c) mlx-mineru.
// OCR text recognition (PP-OCRv6 CTC, pipeline backend) via ONNX Runtime:
// a text-line crop -> text. Faithful to MinerU TextRecognizer + CTCLabelDecode.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

struct RecResult {
  std::string text;
  float score;  // mean per-step CTC confidence of kept characters
};

class TextRecognizer {
 public:
  // onnx_path = ocr_rec.onnx, dict_path = ppocrv6_dict.txt.
  static TextRecognizer load(const std::string& onnx_path, const std::string& dict_path);
  ~TextRecognizer();
  TextRecognizer(TextRecognizer&&) noexcept;
  TextRecognizer& operator=(TextRecognizer&&) noexcept;

  // rgb: w*h*3 RGB8 text-line crop. max_wh_override: the batch-shared max_wh_ratio
  // (MinerU pads a whole rec batch to the widest crop's aspect). <0 -> single-crop
  // behavior (max(w/h, imgW/imgH)).
  RecResult recognize(const std::vector<uint8_t>& rgb, int w, int h,
                      double max_wh_override = -1.0) const;

 private:
  TextRecognizer();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

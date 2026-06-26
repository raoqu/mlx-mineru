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

  // Recognize a group of crops in ONE batched forward. The batch is padded to a fixed size and
  // the width bucketed (rounded up to a multiple of kWbase) so MNN/Metal compiles each shape once
  // and reuses it — that makes the batch ~2x faster per crop (variable shapes would force a
  // recompile per group and lose the win). `max_wh` is the group's shared max_wh_ratio. Returns
  // one RecResult per input crop, in order; degenerate (w/h<=0) crops yield empty.
  std::vector<RecResult> recognize_batch(const std::vector<const std::vector<uint8_t>*>& rgbs,
                                         const std::vector<int>& ws, const std::vector<int>& hs,
                                         double max_wh) const;

 private:
  TextRecognizer();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

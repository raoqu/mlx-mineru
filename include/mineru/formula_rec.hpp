// Copyright (c) mlx-mineru.
// Formula recognition (UniMERNet: Swin encoder + mBART decoder) via ONNX Runtime.
// Formula crop image -> LaTeX. Greedy decode, no KV cache (formulas are short).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

struct FormulaResult {
  std::vector<int> ids;  // generated token ids (drops BOS, keeps trailing EOS)
  std::string latex;     // byte-level decoded, special tokens skipped
};

class FormulaRecognizer {
 public:
  static FormulaRecognizer load(const std::string& encoder_onnx, const std::string& decoder_onnx,
                                const std::string& vocab_path);
  ~FormulaRecognizer();
  FormulaRecognizer(FormulaRecognizer&&) noexcept;
  FormulaRecognizer& operator=(FormulaRecognizer&&) noexcept;

  // gray: H*W normalized-gray pixel map (UniMERNet preprocess output, isolates the model).
  FormulaResult recognize_pixel(const std::vector<float>& gray, int H, int W) const;

  // rgb: w*h*3 RGB8 formula crop -> LaTeX (full preprocess: crop-margin, resize, pad, gray).
  FormulaResult recognize(const std::vector<uint8_t>& rgb, int w, int h) const;

 private:
  FormulaRecognizer();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

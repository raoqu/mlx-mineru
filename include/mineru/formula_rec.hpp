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

// Encoder output for one formula crop (Swin hidden state), used to batch the decoder across a
// page's formulas: encode each crop, then decode_batch() all of them in one MLX pass.
struct FormulaEncoded {
  std::vector<float> hid;  // [n, d_model] row-major
  int n = 0;
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

  // Encoder only: RGB crop -> Swin hidden state (preprocess + encoder). Pair with decode_batch().
  FormulaEncoded encode(const std::vector<uint8_t>& rgb, int w, int h) const;

  // Decode several encoded formulas. With the MLX decoder this is ONE batched GPU pass (B>1);
  // otherwise it decodes them one at a time on ORT. Order matches the input.
  std::vector<FormulaResult> decode_batch(const std::vector<FormulaEncoded>& encs) const;

  struct Impl;  // public type name so the .cpp's free encode/decode helpers can reference it

 private:
  FormulaRecognizer();
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

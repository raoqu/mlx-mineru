// Copyright (c) mlx-mineru.
// Qwen2 byte-level BPE tokenizer — faithful to the HF fast tokenizer shipped with
// the model (Split on the Qwen2 regex -> ByteLevel -> BPE merges). Used to feed
// the Qwen2-VL VLM and decode its output.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mineru {

class Qwen2Tokenizer {
 public:
  // Load from a directory containing vocab.json, merges.txt, tokenizer.json
  // (for added/special tokens), tokenizer_config.json.
  static Qwen2Tokenizer load(const std::string& dir);
  ~Qwen2Tokenizer();
  Qwen2Tokenizer(Qwen2Tokenizer&&) noexcept;
  Qwen2Tokenizer& operator=(Qwen2Tokenizer&&) noexcept;

  // Encode text to token ids. Special/added tokens present literally in `text`
  // are recognized and emitted as their ids (matches HF default add_special_tokens
  // behavior for Qwen2, which adds no automatic BOS/EOS).
  std::vector<int> encode(const std::string& text) const;

  // Decode ids back to text. skip_special drops special-token ids.
  std::string decode(const std::vector<int>& ids, bool skip_special = false) const;

  int token_to_id(const std::string& token) const;  // -1 if absent
  int special_id(const std::string& content) const;  // e.g. "<|im_end|>"; -1 if absent
  int vocab_size() const;

 private:
  Qwen2Tokenizer();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

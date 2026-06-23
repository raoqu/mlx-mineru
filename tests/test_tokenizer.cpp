// Copyright (c) mlx-mineru.
// Phase 4b verification: the C++ Qwen2 BPE tokenizer must match the HF transformers
// tokenizer (scripts/gen_tokenizer_golden.py) — exact token-id parity on encode,
// and exact text on decode (with and without special tokens).
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mineru/tokenizer.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;

static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  std::string tok_dir = (argc > 1) ? argv[1] : "models/MinerU2.5-tokenizer";
  std::string golden_path = (argc > 2) ? argv[2] : "tests/golden/tokenizer.json";

  mineru::Qwen2Tokenizer tok = mineru::Qwen2Tokenizer::load(tok_dir);
  json g = json::parse(read_file(golden_path));

  // Sanity: a couple of known special ids.
  CHECK(tok.special_id("<|im_start|>") == 151644);
  CHECK(tok.special_id("<|im_end|>") == 151645);

  int case_no = 0;
  for (auto& c : g["cases"]) {
    std::string text = c["text"];
    std::vector<int> want = c["ids"].get<std::vector<int>>();
    std::vector<int> got = tok.encode(text);
    bool ok = (got == want);
    CHECK_MSG(ok, "encode mismatch case " + std::to_string(case_no));
    if (!ok) {
      std::cerr << "  text=" << text.substr(0, 60) << "\n  want(" << want.size() << "): ";
      for (size_t k = 0; k < want.size(); ++k) std::cerr << want[k] << " ";
      std::cerr << "\n  got (" << got.size() << "): ";
      for (size_t k = 0; k < got.size(); ++k) std::cerr << got[k] << " ";
      std::cerr << "\n";
    }
    // decode parity.
    CHECK_MSG(tok.decode(want, /*skip_special=*/false) == c["decoded"].get<std::string>(),
              "decode mismatch case " + std::to_string(case_no));
    CHECK_MSG(tok.decode(want, /*skip_special=*/true) == c["decoded_skip"].get<std::string>(),
              "decode(skip_special) mismatch case " + std::to_string(case_no));
    ++case_no;
  }
  return TEST_SUMMARY();
}

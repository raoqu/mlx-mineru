// Copyright (c) mlx-mineru.
// Phase 4d verification: the MLX C++ Qwen2-VL language-model forward matches the
// real transformers model (text-only). Gates on EXACT next-token argmax (greedy
// decode) and the 5-step greedy continuation; top-k logit values are checked
// within tolerance (bf16 matmul ordering differs between torch and MLX).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/qwen2_vl.hpp"
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
  std::string weights = (argc > 1) ? argv[1] : "models/MinerU2.5-tokenizer/model.safetensors";
  std::string golden = (argc > 2) ? argv[2] : "tests/golden/llm.json";

  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(weights);
  json g = json::parse(read_file(golden));

  int case_no = 0;
  for (auto& c : g["cases"]) {
    std::vector<int> ids = c["ids"].get<std::vector<int>>();

    std::vector<float> logits = model.forward_text_logits(ids);
    int argmax = 0;
    for (int i = 1; i < (int)logits.size(); ++i)
      if (logits[i] > logits[argmax]) argmax = i;
    CHECK_MSG(argmax == c["argmax"].get<int>(),
              "case " + std::to_string(case_no) + ": argmax " + std::to_string(argmax) +
                  " != " + std::to_string(c["argmax"].get<int>()));

    // top-k logit values within tolerance (relative, bf16).
    double max_rel = 0;
    for (auto& kv : c["top10"]) {
      int id = kv[0];
      double want = kv[1];
      double got = logits[id];
      double denom = std::max(1.0, std::abs(want));
      max_rel = std::max(max_rel, std::abs(got - want) / denom);
    }
    CHECK_MSG(max_rel < 0.05, "case " + std::to_string(case_no) +
                                  ": top10 logit rel diff " + std::to_string(max_rel));

    // Greedy continuation: at each step compare my argmax to the reference
    // token, driving the prefix from REFERENCE tokens (so each forward is tested
    // against ground truth independently). A mismatch is only tolerated when the
    // reference top-2 gap is a near-tie (bf16 rounding can flip the winner).
    const double kTieGap = 0.25;
    std::vector<int> cur = ids;
    for (auto& step : c["greedy5"]) {
      int want_tok = step["token"];
      double gap = step["gap"];
      int got = model.argmax_next(cur);
      bool ok = (got == want_tok) || (gap < kTieGap);
      CHECK_MSG(ok, "case " + std::to_string(case_no) + ": greedy step got " +
                        std::to_string(got) + " want " + std::to_string(want_tok) +
                        " (gap=" + std::to_string(gap) + ")");
      cur.push_back(want_tok);  // stay aligned with the reference path
    }
    ++case_no;
  }
  return TEST_SUMMARY();
}

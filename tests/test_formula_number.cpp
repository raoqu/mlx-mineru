// Copyright (c) mlx-mineru.
// Pipeline P5: optimize_formula_numbers associates formula_number blocks with adjacent
// interline_equation blocks, appends \tag{N}, drops the merged number (downgrades
// unmatched ones to text). Faithful to MinerU optimize_formula_number_blocks.
#include <iostream>

#include "mineru/pipeline_assemble.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;

static json eq(int idx, const std::string& latex) {
  return {{"type", "interline_equation"}, {"index", idx}, {"bbox", {0, 0, 1, 1}},
          {"lines", json::array({{{"bbox", {0, 0, 1, 1}},
                                  {"spans", json::array({{{"type", "interline_equation"},
                                                          {"content", latex}}})}}})}};
}
static json fn(int idx, const std::string& num) {
  return {{"type", "formula_number"}, {"index", idx}, {"bbox", {0, 0, 1, 1}},
          {"lines", json::array({{{"bbox", {0, 0, 1, 1}},
                                  {"spans", json::array({{{"type", "text"}, {"content", num}}})}}})}};
}
static std::string eq_content(const json& b) {
  return b["lines"][0]["spans"][0]["content"].get<std::string>();
}

int main() {
  // Case 1: number after equation -> tag appended, number dropped (demo1 p2 formula 1).
  {
    json blocks = json::array({eq(9, "Q _ { \\% } = f ( P ) + g ( T )"), fn(10, "(1)")});
    mineru::optimize_formula_numbers(blocks);
    CHECK_MSG(blocks.size() == 1, "formula_number dropped after merge");
    CHECK_MSG(eq_content(blocks[0]) == "Q _ { \\% } = f ( P ) + g ( T )\\tag{1}", "tag appended (prev)");
  }
  // Case 2: number before equation -> tag appended to the next equation.
  {
    json blocks = json::array({fn(1, "(2)"), eq(2, "a + b")});
    mineru::optimize_formula_numbers(blocks);
    CHECK_MSG(blocks.size() == 1, "formula_number dropped (next case)");
    CHECK_MSG(eq_content(blocks[0]) == "a + b\\tag{2}", "tag appended (next)");
  }
  // Case 3: full-width parens normalized.
  {
    json blocks = json::array({eq(1, "x"), fn(2, "\xEF\xBC\x88""3\xEF\xBC\x89")});  // （3）
    mineru::optimize_formula_numbers(blocks);
    CHECK_MSG(eq_content(blocks[0]) == "x\\tag{3}", "full-width parens stripped");
  }
  // Case 4: unmatched formula_number -> downgraded to text, kept.
  {
    json blocks = json::array({{{"type", "text"}, {"index", 1}, {"lines", json::array()}},
                               fn(2, "(9)")});
    mineru::optimize_formula_numbers(blocks);
    CHECK_MSG(blocks.size() == 2, "unmatched number kept");
    CHECK_MSG(blocks[1]["type"] == "text", "unmatched number downgraded to text");
  }
  std::cerr << "formula_number: tag association + downgrade rules match MinerU\n";
  return TEST_SUMMARY();
}

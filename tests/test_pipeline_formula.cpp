// Copyright (c) mlx-mineru.
// Pipeline P5 (visual spans): assemble emits interline_equation blocks carrying the
// display_formula latex. Verified vs MinerU on demo1 p2 (two display formulas). The
// formula-number \tag{N} association is a documented follow-up, so latex is compared with
// any trailing \tag{...} stripped.
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mineru/pipeline_assemble.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
static std::string strip_tag(std::string s) {
  size_t t = s.find("\\tag{");
  return t == std::string::npos ? s : s.substr(0, t);
}
static std::vector<std::string> formula_latex(const json& blocks) {
  std::vector<std::string> out;
  for (const auto& b : blocks)
    if (b.value("type", "") == "interline_equation")
      for (const auto& ln : b.value("lines", json::array()))
        for (const auto& sp : ln.value("spans", json::array()))
          out.push_back(strip_tag(sp.value("content", "")));
  return out;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  json model_page = json::parse(read_file(golden_dir + "/demo1_p2_model.json"));
  json want = json::parse(read_file(golden_dir + "/demo1_p2_middle.json"));
  int page_w = want["page_size"][0], page_h = want["page_size"][1];

  json got = mineru::assemble_page_info(model_page, page_w, page_h, 2);
  auto gf = formula_latex(got["para_blocks"]);
  auto wf = formula_latex(want["para_blocks"]);

  std::cerr << "pipeline_formula: " << gf.size() << " interline_equation blocks vs golden "
            << wf.size() << "\n";
  CHECK_MSG(gf.size() == wf.size(), "formula block count");
  CHECK_MSG(gf.size() >= 1, "at least one formula");
  int ok = 0;
  size_t n = std::min(gf.size(), wf.size());
  for (size_t i = 0; i < n; ++i) {
    if (gf[i] == wf[i]) ++ok;
    else std::cerr << "  formula " << i << ":\n    got : " << gf[i] << "\n    want: " << wf[i] << "\n";
  }
  CHECK_MSG(ok == (int)n, "formula latex matches MinerU (modulo \\tag)");
  for (size_t i = 0; i < n; ++i) std::cerr << "  ✓ " << gf[i] << "\n";
  return TEST_SUMMARY();
}

// Copyright (c) mlx-mineru.
// Pipeline P5 (visual spans): assemble emits table blocks {type:table, blocks:[table_body
// (html span)]} carrying the recognized HTML. Verified vs MinerU on demo1 p5 (two tables).
// Caption/footnote nesting (find_best_visual_parent) is a documented follow-up.
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

// Extract table_body html strings from para_blocks (handles two-layer table blocks).
static std::vector<std::string> table_html(const json& blocks) {
  std::vector<std::string> out;
  for (const auto& b : blocks) {
    if (b.value("type", "") != "table") continue;
    for (const auto& sub : b.value("blocks", json::array()))
      if (sub.value("type", "") == "table_body")
        for (const auto& ln : sub.value("lines", json::array()))
          for (const auto& sp : ln.value("spans", json::array()))
            if (sp.contains("html")) out.push_back(sp["html"].get<std::string>());
  }
  return out;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  json model_page = json::parse(read_file(golden_dir + "/demo1_p5_model.json"));
  json want = json::parse(read_file(golden_dir + "/demo1_p5_middle.json"));
  int page_w = want["page_size"][0], page_h = want["page_size"][1];

  json got = mineru::assemble_page_info(model_page, page_w, page_h, 5);
  auto gh = table_html(got["para_blocks"]);
  auto wh = table_html(want["para_blocks"]);

  std::cerr << "pipeline_table: " << gh.size() << " table bodies vs golden " << wh.size() << "\n";
  CHECK_MSG(gh.size() == wh.size(), "table count");
  CHECK_MSG(gh.size() >= 1, "at least one table");
  int ok = 0;
  size_t n = std::min(gh.size(), wh.size());
  for (size_t i = 0; i < n; ++i) {
    if (gh[i] == wh[i]) ++ok;
    else std::cerr << "  table " << i << " html differs:\n    got : " << gh[i].substr(0, 120)
                   << "\n    want: " << wh[i].substr(0, 120) << "\n";
  }
  CHECK_MSG(ok == (int)n, "table_body html matches MinerU");
  std::cerr << "pipeline_table: " << ok << "/" << n << " table HTML match\n";
  return TEST_SUMMARY();
}

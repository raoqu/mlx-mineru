// Copyright (c) mlx-mineru.
// Phase 4e verification: the C++ layout-output parser matches mineru-vl-utils
// parse_layout_output (regex + bbox conversion + table-internal filter).
#include <cmath>
#include <fstream>
#include <sstream>

#include "mineru/vlm_layout.hpp"
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
  std::string golden = (argc > 1) ? argv[1] : "tests/golden/layout.json";
  json g = json::parse(read_file(golden));

  int ci = 0;
  for (auto& s : g["samples"]) {
    std::vector<mineru::ContentBlock> blocks = mineru::parse_layout_output(s["output"]);
    auto& want = s["blocks"];
    CHECK_MSG(blocks.size() == want.size(),
              "sample " + std::to_string(ci) + ": block count " + std::to_string(blocks.size()) +
                  " != " + std::to_string(want.size()));
    for (size_t i = 0; i < blocks.size() && i < want.size(); ++i) {
      CHECK_MSG(blocks[i].type == want[i]["type"].get<std::string>(),
                "sample " + std::to_string(ci) + " block " + std::to_string(i) + " type");
      for (int k = 0; k < 4; ++k)
        CHECK(std::abs(blocks[i].bbox[k] - want[i]["bbox"][k].get<double>()) < 1e-9);
      bool want_angle = !want[i]["angle"].is_null();
      CHECK_MSG(blocks[i].angle.has_value() == want_angle,
                "sample " + std::to_string(ci) + " block " + std::to_string(i) + " angle presence");
      if (want_angle && blocks[i].angle.has_value())
        CHECK(*blocks[i].angle == want[i]["angle"].get<int>());
      CHECK(blocks[i].merge_prev == want[i]["merge_prev"].get<bool>());
    }
    ++ci;
  }
  return TEST_SUMMARY();
}

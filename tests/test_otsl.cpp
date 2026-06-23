// Copyright (c) mlx-mineru.
// Verify the OTSL->HTML port matches mineru-vl-utils convert_otsl_to_html.
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/otsl.hpp"
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
  std::string golden = (argc > 1) ? argv[1] : "tests/golden/otsl.json";
  json g = json::parse(read_file(golden));
  int i = 0;
  for (auto& s : g["samples"]) {
    std::string got = mineru::convert_otsl_to_html(s["otsl"].get<std::string>());
    std::string want = s["html"].get<std::string>();
    bool ok = (got == want);
    CHECK_MSG(ok, "otsl sample " + std::to_string(i) + " mismatch");
    if (!ok) std::cerr << "  otsl=" << s["otsl"].get<std::string>() << "\n  got =" << got
                       << "\n  want=" << want << "\n";
    ++i;
  }
  return TEST_SUMMARY();
}

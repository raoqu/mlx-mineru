// Copyright (c) mlx-mineru.
// Verify text/equation post_process ports match mineru-vl-utils.
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include "mineru/post_process.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
namespace pp = mineru::pp;

static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  std::string golden = (argc > 1) ? argv[1] : "tests/golden/post_process.json";
  json g = json::parse(read_file(golden));

  std::map<std::string, std::function<std::string(const std::string&)>> fns = {
      {"display_to_inline", pp::try_convert_display_to_inline},
      {"macro_spacing", pp::try_fix_macro_spacing_in_markdown},
      {"move_underscores", pp::try_move_underscores_outside},
      {"eq_delim", pp::try_fix_equation_delimiters},
      {"unbalanced", pp::try_fix_unbalanced_braces}};

  for (auto& [key, fn] : fns) {
    int i = 0;
    for (auto& c : g[key]) {
      std::string got = fn(c["in"].get<std::string>());
      std::string want = c["out"].get<std::string>();
      bool ok = (got == want);
      CHECK_MSG(ok, key + " case " + std::to_string(i) + " mismatch");
      if (!ok) std::cerr << "  in  =" << c["in"].get<std::string>() << "\n  got =" << got
                         << "\n  want=" << want << "\n";
      ++i;
    }
  }
  return TEST_SUMMARY();
}

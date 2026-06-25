// Copyright (c) mlx-mineru.
// cross_page_table_merge == MinerU merge_table() golden. Each golden case is
// {name, input(pdf_info), expected(pdf_info after merge_table)}; we run our port on a copy
// of input and require deep JSON equality with expected.
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/table_merge.hpp"
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
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  json cases = json::parse(read_file(golden_dir + "/table_merge_golden.json"));

  for (const json& c : cases) {
    std::string name = c["name"];
    json got = c["input"];  // deep copy
    try {
      mineru::cross_page_table_merge(got);
    } catch (const std::exception& e) {
      std::cerr << "  [" << name << "] THREW: " << e.what() << "\n";
      CHECK_MSG(false, name);
      continue;
    }
    const json& want = c["expected"];
    bool ok = (got == want);
    CHECK_MSG(ok, name);
    if (!ok) {
      std::cerr << "  [" << name << "] mismatch\n";
      // pinpoint first differing page/block html for quick triage
      for (size_t p = 0; p < want.size(); ++p) {
        if (p < got.size() && got[p] != want[p])
          std::cerr << "   page " << p << "\n    got : " << got[p].dump()
                    << "\n    want: " << want[p].dump() << "\n";
      }
    } else {
      std::cerr << "table_merge " << name << ": ok\n";
    }
  }
  return TEST_SUMMARY();
}

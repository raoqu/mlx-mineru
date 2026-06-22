// Copyright (c) mlx-mineru.
// Phase 1 verification: the C++ union_make output must match golden output
// produced by the REAL MinerU Python union_make (scripts/gen_golden.py).
//   - Markdown modes: exact string equality.
//   - content_list modes: semantic JSON equality (key order independent).
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mineru/enums.hpp"
#include "mineru/mkcontent.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;

static std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  std::string data_dir = (argc > 1) ? argv[1] : "tests";
  std::string fixture = data_dir + "/data/sample_middle.json";
  std::string golden = data_dir + "/golden";

  json mj = json::parse(read_file(fixture));
  const json& pdf_info = mj["pdf_info"];
  const std::string bucket = "images";

  // Markdown: exact string match.
  {
    std::string got = mineru::union_make(pdf_info, mineru::make_mode::kMmMd, bucket).get<std::string>();
    std::string want = read_file(golden + "/sample.mm.md");
    CHECK_MSG(got == want, "MM_MD mismatch");
    if (got != want) std::cerr << "--- MM_MD got ---\n" << got << "\n--- want ---\n" << want << "\n";
  }
  {
    std::string got = mineru::union_make(pdf_info, mineru::make_mode::kNlpMd, bucket).get<std::string>();
    std::string want = read_file(golden + "/sample.nlp.md");
    CHECK_MSG(got == want, "NLP_MD mismatch");
    if (got != want) std::cerr << "--- NLP_MD got ---\n" << got << "\n--- want ---\n" << want << "\n";
  }
  // content_list v1/v2: semantic JSON equality.
  {
    json got = mineru::union_make(pdf_info, mineru::make_mode::kContentList, bucket);
    json want = json::parse(read_file(golden + "/sample.content_list.json"));
    CHECK_MSG(got == want, "CONTENT_LIST mismatch");
    if (got != want) std::cerr << "--- CL got ---\n" << got.dump(2) << "\n--- want ---\n" << want.dump(2) << "\n";
  }
  {
    json got = mineru::union_make(pdf_info, mineru::make_mode::kContentListV2, bucket);
    json want = json::parse(read_file(golden + "/sample.content_list_v2.json"));
    CHECK_MSG(got == want, "CONTENT_LIST_V2 mismatch");
    if (got != want) std::cerr << "--- CLV2 got ---\n" << got.dump(2) << "\n--- want ---\n" << want.dump(2) << "\n";
  }
  return TEST_SUMMARY();
}

// Copyright (c) mlx-mineru.
// Pipeline P5: C++ assemble_page_info(model_list page) == MinerU middle_json page_info
// (text/title path) on a.pdf p0.
#include <fstream>
#include <iostream>
#include <sstream>

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

// Span text content is filled by MinerU's separate post-OCR step (re-running OCR rec on
// each span crop), not by the assembly -- the model_list ocr_text boxes ship empty text.
// Blank span content on both sides so the test verifies the assembly STRUCTURE exactly.
static void blank_content(json& blocks) {
  for (auto& b : blocks) {
    if (!b.contains("lines")) continue;
    for (auto& ln : b["lines"]) {
      if (!ln.contains("spans")) continue;
      for (auto& sp : ln["spans"])
        if (sp.contains("content")) sp["content"] = "";
    }
  }
}

// Compare two block arrays field-by-field, reporting the first divergence.
static bool cmp_blocks(const char* tag, const json& got, const json& want) {
  if (got.size() != want.size()) {
    std::cerr << "  [" << tag << "] count " << got.size() << " != " << want.size() << "\n";
    return false;
  }
  bool ok = true;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      ok = false;
      std::cerr << "  [" << tag << "] block " << i << " differs:\n    got : "
                << got[i].dump() << "\n    want: " << want[i].dump() << "\n";
    }
  }
  return ok;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  json model_page = json::parse(read_file(golden_dir + "/pipeline_p0_model.json"));
  json want = json::parse(read_file(golden_dir + "/pipeline_p0_middle.json"));

  int page_w = want["page_size"][0], page_h = want["page_size"][1];
  json got = mineru::assemble_page_info(model_page, page_w, page_h, want.value("page_idx", 0));

  // Compare structure only (post-OCR fills span text separately).
  blank_content(got["para_blocks"]);
  blank_content(got["preproc_blocks"]);
  blank_content(want["para_blocks"]);
  blank_content(want["preproc_blocks"]);

  bool para_ok = cmp_blocks("para_blocks", got["para_blocks"], want["para_blocks"]);
  bool pre_ok = cmp_blocks("preproc_blocks", got["preproc_blocks"], want["preproc_blocks"]);
  bool disc_ok = (got["discarded_blocks"] == want["discarded_blocks"]);
  bool size_ok = (got["page_size"] == want["page_size"]);

  CHECK_MSG(para_ok, "para_blocks match MinerU");
  CHECK_MSG(pre_ok, "preproc_blocks match MinerU");
  CHECK_MSG(disc_ok, "discarded_blocks match");
  CHECK_MSG(size_ok, "page_size match");
  std::cerr << "pipeline_assemble: " << got["para_blocks"].size() << " para_blocks, "
            << got["discarded_blocks"].size() << " discarded — "
            << (para_ok && pre_ok ? "match" : "MISMATCH") << " MinerU\n";
  return TEST_SUMMARY();
}

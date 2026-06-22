// Copyright (c) mlx-mineru.
// Phase 0 verification: parse a middle.json fixture into the typed model and
// serialize it back; the result must be SEMANTICALLY EQUAL to the input
// (key order may differ, values/structure must not). This proves the typed
// model is field-level lossless, including unknown/future keys.
#include <fstream>
#include <sstream>
#include <string>

#include "mineru/enums.hpp"
#include "mineru/middle_json.hpp"
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
  std::string fixture = (argc > 1) ? argv[1] : "tests/data/sample_middle.json";
  std::string text = read_file(fixture);
  CHECK_MSG(!text.empty(), "could not read fixture: " + fixture);
  if (text.empty()) return TEST_SUMMARY();

  // 1. Reference parse (untyped).
  json reference = json::parse(text);

  // 2. Parse into typed model, then serialize back to untyped json.
  mineru::MiddleJson doc = mineru::parse_middle_json(text);
  json roundtrip = json(doc);

  // 3. Semantic equality (order-independent for objects).
  CHECK_MSG(reference == roundtrip,
            "typed round-trip not semantically equal to input");

  // 4. Spot-check that typed fields were actually populated (not all dumped to extra).
  CHECK(doc.pdf_info.size() == 1);
  CHECK(doc.backend.has_value() && *doc.backend == "vlm");
  CHECK(doc.version_name.has_value() && *doc.version_name == mineru::kMineruVersion);
  if (!doc.pdf_info.empty()) {
    const auto& page = doc.pdf_info[0];
    CHECK(page.page_idx.has_value() && *page.page_idx == 0);
    CHECK(page.para_blocks.size() == 7);
    CHECK(page.discarded_blocks.size() == 1);
    if (!page.para_blocks.empty()) {
      const auto& title = page.para_blocks[0];
      CHECK(title.type == mineru::block_type::kTitle);
      CHECK(title.level.has_value() && *title.level == 1);
      CHECK(title.lines.has_value() && title.lines->size() == 1);
    }
    // Unknown future field preserved on the page.
    CHECK_MSG(page.extra.contains("an_unknown_future_field"),
              "page-level unknown field was dropped");
  }
  // Unknown top-level field preserved.
  CHECK_MSG(doc.extra.contains("_an_unknown_top_level_field"),
            "top-level unknown field was dropped");

  // 5. Stable: parse(dump(parse(x))) == parse(x).
  std::string dumped = mineru::dump_middle_json(doc);
  CHECK(json::parse(dumped) == reference);

  return TEST_SUMMARY();
}

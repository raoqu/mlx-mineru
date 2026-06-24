// Copyright (c) mlx-mineru.
// Pipeline P2: C++ SLANet+ structurer == MinerU TableStructurer + adapt_slanet_plus.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/table_rec.hpp"
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
  std::string onnx = (argc > 2) ? argv[2] : "models/pipeline/TabRec/SlanetPlus/slanet-plus.onnx";
  std::string dict = (argc > 3) ? argv[3] : "models/pipeline/TabRec/SlanetPlus/table_structure_dict.txt";

  json g = json::parse(read_file(golden_dir + "/table_struct.json"));
  int sw = g["src_w"], sh = g["src_h"];
  std::string raw = read_file(golden_dir + "/" + g["input"].get<std::string>());
  CHECK_MSG((int)raw.size() == sw * sh * 3, "table rgb size");
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  mineru::TableRecognizer tr = mineru::TableRecognizer::load(onnx, dict);
  auto ts = tr.recognize_structure(rgb, sw, sh);

  auto want_tokens = g["structure"].get<std::vector<std::string>>();
  bool tok_ok = (ts.tokens == want_tokens);
  CHECK_MSG(tok_ok, "structure tokens exact");
  if (!tok_ok) {
    std::cerr << "  got " << ts.tokens.size() << " tokens, want " << want_tokens.size() << "\n";
    for (size_t i = 0; i < std::min(ts.tokens.size(), want_tokens.size()); ++i)
      if (ts.tokens[i] != want_tokens[i]) {
        std::cerr << "  first diff @" << i << ": got '" << ts.tokens[i] << "' want '"
                  << want_tokens[i] << "'\n"; break;
      }
  }

  auto& wc = g["cells"];
  CHECK_MSG(ts.cells.size() == wc.size(), "cell count");
  float maxd = 0;
  for (size_t i = 0; i < std::min(ts.cells.size(), wc.size()); ++i)
    for (int k = 0; k < 8; ++k)
      maxd = std::max(maxd, std::abs(ts.cells[i][k] - wc[i][k].get<float>()));
  CHECK_MSG(maxd <= 1.0f, "cell bboxes within 1px");
  std::cerr << "table struct: " << ts.tokens.size() << " tokens, " << ts.cells.size()
            << " cells, max cell delta " << maxd << "px\n";

  // --- matcher path: structure + OCR -> HTML, exact vs MinerU TableMatch ---
  std::string mpath = golden_dir + "/table_match.json";
  std::ifstream mf(mpath);
  if (mf.good()) {
    json gm = json::parse(read_file(mpath));
    std::vector<mineru::TableOcrItem> ocr;
    for (auto& it : gm["ocr"]) {
      mineru::TableOcrItem o;
      for (int k = 0; k < 4; ++k) o.box[k] = {it["box"][k][0].get<float>(), it["box"][k][1].get<float>()};
      o.text = it["text"].get<std::string>();
      o.score = it["score"].get<float>();
      ocr.push_back(o);
    }
    std::string html = tr.recognize_html(rgb, sw, sh, ocr);
    std::string want_html = gm["pred_html"].get<std::string>();
    bool html_ok = (html == want_html);
    CHECK_MSG(html_ok, "table html exact");
    if (!html_ok) std::cerr << "  got : " << html << "\n  want: " << want_html << "\n";
    std::cerr << "table html: " << (html_ok ? "matches" : "MISMATCH") << " MinerU\n  " << html << "\n";
  }
  return TEST_SUMMARY();
}

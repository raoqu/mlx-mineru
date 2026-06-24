// Copyright (c) mlx-mineru.
// Pipeline P2: C++ TableClassifier == Python golden (same resize/normalize/onnx).
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/table_cls.hpp"
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
  std::string onnx = (argc > 1) ? argv[1] : "models/pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx";
  std::string golden_dir = (argc > 2) ? argv[2] : "tests/golden";

  json g = json::parse(read_file(golden_dir + "/tabcls.json"));
  int size = g["size"];
  std::string raw = read_file(golden_dir + "/" + g["input"].get<std::string>());
  CHECK_MSG((int)raw.size() == size * size * 3, "input rgb size");
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  mineru::TableClassifier clf = mineru::TableClassifier::load(onnx);
  auto r = clf.classify(rgb, size, size);

  CHECK_MSG(r.cls_id == g["cls_id"].get<int>(), "cls_id");
  CHECK_MSG(r.label == g["label"].get<std::string>(), "label");
  CHECK_MSG(std::abs(r.probs[0] - g["probs"][0].get<float>()) < 1e-3 &&
                std::abs(r.probs[1] - g["probs"][1].get<float>()) < 1e-3,
            "probs");
  std::cerr << "tab_cls: " << r.label << " " << r.score << " (probs " << r.probs[0] << ","
            << r.probs[1] << ") matches golden\n";
  return TEST_SUMMARY();
}

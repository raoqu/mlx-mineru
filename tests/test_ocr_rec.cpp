// Copyright (c) mlx-mineru.
// Pipeline P3: C++ TextRecognizer == Python golden (same resize/normalize/CTC).
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/ocr_rec.hpp"
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
  std::string onnx = (argc > 1) ? argv[1] : "models/pipeline/OCR/ocr_rec.onnx";
  std::string dict = (argc > 2) ? argv[2] : "models/pipeline/OCR/ppocrv6_dict.txt";
  std::string golden_dir = (argc > 3) ? argv[3] : "tests/golden";

  json g = json::parse(read_file(golden_dir + "/ocr_rec.json"));
  int w = g["w"], h = g["h"];
  std::string raw = read_file(golden_dir + "/" + g["input"].get<std::string>());
  CHECK_MSG((int)raw.size() == w * h * 3, "input rgb size");
  std::vector<uint8_t> rgb(raw.begin(), raw.end());

  mineru::TextRecognizer rec = mineru::TextRecognizer::load(onnx, dict);
  auto r = rec.recognize(rgb, w, h);

  bool text_ok = (r.text == g["text"].get<std::string>());
  CHECK_MSG(text_ok, "text mismatch");
  if (!text_ok) std::cerr << "  got : " << r.text << "\n  want: " << g["text"].get<std::string>() << "\n";
  CHECK_MSG(std::abs(r.score - g["score"].get<float>()) < 1e-3, "score");
  std::cerr << "ocr_rec: \"" << r.text << "\" (score " << r.score << ") matches golden\n";
  return TEST_SUMMARY();
}

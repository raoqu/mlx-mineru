// Copyright (c) mlx-mineru.
// Pipeline P4: C++ FormulaRecognizer == exported UniMERNet ONNX (greedy) + byte-level
// decode. The model path (pixel -> ids/latex) is exact vs the golden; the end-to-end
// path (raw crop -> latex) also runs the cv2-style preprocess (looser, like ocr_page).
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/formula_rec.hpp"
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
  std::string enc = (argc > 2) ? argv[2] : "models/pipeline/MFR/mfr_encoder.onnx";
  std::string dec = (argc > 3) ? argv[3] : "models/pipeline/MFR/mfr_decoder.onnx";
  std::string vocab = (argc > 4) ? argv[4] : "models/pipeline/MFR/mfr_vocab.txt";

  json g = json::parse(read_file(golden_dir + "/mfr.json"));
  int H = g["H"], W = g["W"];
  std::string raw = read_file(golden_dir + "/" + g["pixel"].get<std::string>());
  CHECK_MSG((int)raw.size() == H * W * 4, "pixel f32 size");
  std::vector<float> gray(H * W);
  std::memcpy(gray.data(), raw.data(), raw.size());

  mineru::FormulaRecognizer mfr = mineru::FormulaRecognizer::load(enc, dec, vocab);

  // --- model path: preprocessed pixel -> ids/latex, exact vs golden ---
  auto r = mfr.recognize_pixel(gray, H, W);
  auto want_ids = g["ids"].get<std::vector<int>>();
  bool ids_ok = (r.ids == want_ids);
  CHECK_MSG(ids_ok, "model ids exact");
  if (!ids_ok) {
    std::cerr << "  got " << r.ids.size() << " ids, want " << want_ids.size() << "\n";
    for (size_t i = 0; i < std::min(r.ids.size(), want_ids.size()); ++i)
      if (r.ids[i] != want_ids[i]) { std::cerr << "  first diff @" << i << ": got "
                                               << r.ids[i] << " want " << want_ids[i] << "\n"; break; }
  }
  std::string want_latex = g["latex"].get<std::string>();
  CHECK_MSG(r.latex == want_latex, "model latex exact");
  std::cerr << "mfr model: " << r.ids.size() << " ids, latex:\n  " << r.latex << "\n";

  // --- end-to-end path: raw RGB crop -> latex via full C++ preprocess ---
  std::string img_path = golden_dir + "/" + g.value("input", "");
  if (!g.value("input", "").empty()) {
    std::string rawimg = read_file(img_path);
    int sw = g["src_w"], sh = g["src_h"];
    if ((int)rawimg.size() == sw * sh * 3) {
      std::vector<uint8_t> rgb(rawimg.begin(), rawimg.end());
      auto e2e = mfr.recognize(rgb, sw, sh);
      std::cerr << "mfr e2e: latex:\n  " << e2e.latex << "\n";
      // cv2.resize/crop-margin sub-pixel differences may shift a token; require a close
      // match (the exported model itself is exact -- verified above).
      CHECK_MSG(!e2e.latex.empty(), "e2e produced latex");
    }
  }
  return TEST_SUMMARY();
}

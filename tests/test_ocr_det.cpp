// Copyright (c) mlx-mineru.
// Pipeline P3: C++ db_postprocess == MinerU DBPostProcess on the SAME prob-map.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mineru/ocr_det.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;
static std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Match by axis-aligned bbox (min/max corner) within a small pixel tolerance —
// robust to corner-ordering and float-rounding (np.round vs lround) differences.
struct AABB { int x0, y0, x1, y1; };
static AABB aabb(const std::array<std::array<int, 2>, 4>& p) {
  AABB b{p[0][0], p[0][1], p[0][0], p[0][1]};
  for (auto& q : p) {
    b.x0 = std::min(b.x0, q[0]); b.y0 = std::min(b.y0, q[1]);
    b.x1 = std::max(b.x1, q[0]); b.y1 = std::max(b.y1, q[1]);
  }
  return b;
}

// Greedy AABB match count between produced boxes and golden AABBs (tol px).
static int match_count(const std::vector<mineru::DetBox>& boxes, const std::vector<AABB>& want,
                       int tol, const char* tag) {
  int matched = 0;
  std::vector<bool> used(want.size(), false);
  for (auto& db : boxes) {
    AABB a = aabb(db.pts);
    int best = -1, bestd = 1e9;
    for (size_t j = 0; j < want.size(); ++j) {
      if (used[j]) continue;
      int d = std::abs(a.x0 - want[j].x0) + std::abs(a.y0 - want[j].y0) +
              std::abs(a.x1 - want[j].x1) + std::abs(a.y1 - want[j].y1);
      if (d < bestd) { bestd = d; best = (int)j; }
    }
    if (best >= 0 && bestd <= 4 * tol) { used[best] = true; ++matched; }
    else std::cerr << "  [" << tag << "] unmatched box: [" << a.x0 << "," << a.y0 << ","
                   << a.x1 << "," << a.y1 << "] (nearest d=" << bestd << ")\n";
  }
  for (size_t j = 0; j < want.size(); ++j)
    if (!used[j]) std::cerr << "  [" << tag << "] missed golden: [" << want[j].x0 << ","
                            << want[j].y0 << "," << want[j].x1 << "," << want[j].y1 << "]\n";
  return matched;
}

int main(int argc, char** argv) {
  std::string golden_dir = (argc > 1) ? argv[1] : "tests/golden";
  std::string onnx = (argc > 2) ? argv[2] : "";
  json g = json::parse(read_file(golden_dir + "/ocr_det.json"));
  int Hd = g["Hd"], Wd = g["Wd"];
  std::array<double, 4> shape;
  for (int i = 0; i < 4; ++i) shape[i] = g["shape_list"][i];

  std::string raw = read_file(golden_dir + "/" + g["probmap"].get<std::string>());
  CHECK_MSG((int)raw.size() == Hd * Wd * 4, "probmap size");
  std::vector<float> pred(Hd * Wd);
  std::memcpy(pred.data(), raw.data(), raw.size());

  auto boxes = mineru::db_postprocess(pred, Hd, Wd, shape, g["thresh"], g["box_thresh"],
                                      g["unclip_ratio"]);

  // Build golden AABBs.
  std::vector<AABB> want;
  for (auto& b : g["boxes"]) {
    std::array<std::array<int, 2>, 4> p;
    for (int i = 0; i < 4; ++i) { p[i] = {b[i][0].get<int>(), b[i][1].get<int>()}; }
    want.push_back(aabb(p));
  }
  std::cerr << "ocr_det: C++ db_postprocess " << boxes.size() << " boxes vs golden "
            << want.size() << "\n";

  const int TOL = 2;
  int matched = match_count(boxes, want, TOL, "postproc");
  CHECK_MSG(boxes.size() == want.size(), "db_postprocess box count");
  CHECK_MSG(matched == (int)want.size(), "db_postprocess boxes matched within tol");
  std::cerr << "ocr_det: postproc " << matched << "/" << want.size() << " match (tol "
            << TOL << "px)\n";

  // End-to-end: full C++ detect() (resize + normalize + onnx + db_postprocess) on the
  // saved source render, when the onnx model and image fixture are available.
  std::string img_path = golden_dir + "/" + g.value("input", "");
  if (!onnx.empty() && !g.value("input", "").empty()) {
    std::string raw_img = read_file(img_path);
    int sw = g["src_w"], sh = g["src_h"];
    if ((int)raw_img.size() == sw * sh * 3) {
      std::vector<uint8_t> rgb(raw_img.begin(), raw_img.end());
      mineru::TextDetector det = mineru::TextDetector::load(onnx);
      auto e2e = det.detect(rgb, sw, sh);
      std::cerr << "ocr_det: C++ end-to-end " << e2e.size() << " boxes vs golden "
                << want.size() << "\n";
      // cv2.resize vs our bilinear differ slightly -> allow a slightly looser tol.
      const int ETOL = 4;
      int em = match_count(e2e, want, ETOL, "e2e");
      CHECK_MSG(em >= (int)want.size() - 1, "e2e boxes matched within tol");
      std::cerr << "ocr_det: e2e " << em << "/" << want.size() << " match (tol "
                << ETOL << "px)\n";
    } else {
      std::cerr << "ocr_det: skipping e2e (image size mismatch)\n";
    }
  } else {
    std::cerr << "ocr_det: skipping e2e (no onnx/image fixture)\n";
  }
  return TEST_SUMMARY();
}

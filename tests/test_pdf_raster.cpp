// Copyright (c) mlx-mineru.
// Phase 3 verification: C++ pdfium rasterization must match MinerU's page_to_image.
// Gates on EXACT image dimensions / scale / page size (the scale+ceil+cap contract);
// content correctness checked via ink-ratio and mean-luminance within tolerance
// (robust to anti-aliasing differences between pdfium builds 7891 vs 7906).
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mineru/pdf.hpp"
#include "nlohmann/json.hpp"
#include "test_util.hpp"

using nlohmann::json;

static std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void metrics(const mineru::PageImage& img, double& ink_ratio, double& mean_lum) {
  size_t n = static_cast<size_t>(img.width) * img.height;
  long long ink = 0;
  long long sum = 0;
  for (size_t k = 0; k < img.rgb.size(); k += 3) {
    int lum = (img.rgb[k] * 299 + img.rgb[k + 1] * 587 + img.rgb[k + 2] * 114) / 1000;
    sum += lum;
    if (lum < 128) ++ink;
  }
  ink_ratio = static_cast<double>(ink) / n;
  mean_lum = static_cast<double>(sum) / n;
}

int main(int argc, char** argv) {
  std::string golden_path = (argc > 1) ? argv[1] : "tests/golden/pdf_raster.json";
  std::string pdf_dir = (argc > 2) ? argv[2] : "";  // dir containing demo PDFs

  json g = json::parse(read_file(golden_path));
  int dpi = g["dpi"];
  int max_edge = g["max_edge"];

  for (auto& [name, fileinfo] : g["files"].items()) {
    std::string path = pdf_dir + "/" + name;
    mineru::PdfDocument doc = mineru::PdfDocument::open_file(path);
    CHECK_MSG(doc.page_count() == fileinfo["page_count"].get<int>(), name + ": page_count");

    for (auto& pg : fileinfo["pages"]) {
      int idx = pg["page_idx"];
      mineru::PageImage img = doc.render_page(idx, dpi, max_edge);
      std::string tag = name + " p" + std::to_string(idx);

      CHECK_MSG(img.width == pg["img_w"].get<int>(), tag + ": img_w " + std::to_string(img.width) +
                                                         " != " + std::to_string(pg["img_w"].get<int>()));
      CHECK_MSG(img.height == pg["img_h"].get<int>(), tag + ": img_h");
      CHECK_MSG(std::abs(img.scale - pg["scale"].get<double>()) < 1e-6, tag + ": scale");
      CHECK_MSG(std::abs(img.width_pt - pg["width_pt"].get<double>()) < 1e-3, tag + ": width_pt");
      CHECK_MSG(std::abs(img.height_pt - pg["height_pt"].get<double>()) < 1e-3, tag + ": height_pt");

      double ink_ratio, mean_lum;
      metrics(img, ink_ratio, mean_lum);
      // Content tolerance absorbs anti-aliasing differences between the C++ pdfium
      // build (7906) and pypdfium2's bundled build (7891); still tight enough to
      // catch blank/wrong-page/channel-swapped renders (those differ by >>this).
      double dink = std::abs(ink_ratio - pg["ink_ratio"].get<double>());
      double dlum = std::abs(mean_lum - pg["mean_lum"].get<double>());
      CHECK_MSG(dink < 0.02, tag + ": ink_ratio diff " + std::to_string(dink) +
                                 " (got " + std::to_string(ink_ratio) + ")");
      CHECK_MSG(dlum < 5.0, tag + ": mean_lum diff " + std::to_string(dlum) +
                                " (got " + std::to_string(mean_lum) + ")");
    }
  }
  return TEST_SUMMARY();
}

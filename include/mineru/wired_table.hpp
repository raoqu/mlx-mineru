// Copyright (c) mlx-mineru.
// Wired (bordered) table structure recognition (UNet, pipeline backend): table crop ->
// line segmentation -> cell polygons -> logical grid -> HTML. Faithful port of MinerU
// unet_table (TSRUnet + TableRecover + plot_html_table). Built up in stages.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

class WiredTableRecognizer {
 public:
  static WiredTableRecognizer load(const std::string& onnx);
  ~WiredTableRecognizer();
  WiredTableRecognizer(WiredTableRecognizer&&) noexcept;
  WiredTableRecognizer& operator=(WiredTableRecognizer&&) noexcept;

  // Stage 1: run the UNet on a table crop (RGB w*h*3) -> line segmentation map.
  // Output is nh*nw row-major; values 0=background, 1=horizontal line, 2=vertical line.
  // nh/nw are the resized model dims (longest side = 1024, aspect kept).
  std::vector<uint8_t> segment(const std::vector<uint8_t>& rgb, int w, int h, int& nh,
                               int& nw) const;

  // Stage 2: segmentation -> cell polygons (4 corners each, [x0,y0,..,x3,y3]) in the crop's
  // pixel space. Faithful to TSRUnet.postprocess (line extract + draw + region boxes), for
  // the non-rotated case.
  std::vector<std::array<float, 8>> cell_polygons(const std::vector<uint8_t>& rgb, int w,
                                                  int h) const;

  // Stage 3: cells -> logical grid. polygons sorted in reading order ([tl,bl,br,tr] per
  // cell); logic[i] = [row_start, row_end, col_start, col_end]. Faithful to TSRUnet.__call__
  // corner ordering + TableRecover.
  struct Structure {
    std::vector<std::array<float, 8>> polygons;
    std::vector<std::array<int, 4>> logic;
  };
  Structure recognize_structure(const std::vector<uint8_t>& rgb, int w, int h) const;

 private:
  WiredTableRecognizer();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

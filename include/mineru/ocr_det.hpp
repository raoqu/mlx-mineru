// Copyright (c) mlx-mineru.
// OCR text detection (PP-OCRv6 DBNet, pipeline backend): probability map ->
// text-line quad boxes. Faithful port of MinerU DBPostProcess (box_type="quad"):
// binarize -> contours -> minAreaRect -> box-score -> unclip -> scale.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mineru {

struct DetBox {
  std::array<std::array<int, 2>, 4> pts;  // 4 corners (x,y), ordered like MinerU
  float score;
};

// Run the DB post-process on a probability map (Hd x Wd, row-major float in [0,1]).
// shape = {src_h, src_w, ratio_h, ratio_w}; boxes are scaled to src_w x src_h.
// Params default to MinerU's det settings.
std::vector<DetBox> db_postprocess(const std::vector<float>& probmap, int Hd, int Wd,
                                   const std::array<double, 4>& shape, float thresh = 0.3f,
                                   float box_thresh = 0.6f, float unclip_ratio = 1.5f);

}  // namespace mineru

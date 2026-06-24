// Copyright (c) mlx-mineru.
// PP-DocLayoutV2 layout detector (pipeline backend) via ONNX Runtime.
// page image -> region boxes {bbox, class, score}. Faithful to MinerU's core
// post-processing (box decode -> scale -> topk -> conf -> clip).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

struct LayoutBox {
  std::array<int, 4> bbox;  // x0,y0,x1,y1 in the target image's pixels
  int cls_id;
  std::string label;
  float score;
  int index = 0;  // 1-based reading order (PP-DocLayoutV2 reading-order head)
};

class LayoutDetector {
 public:
  // model_dir must contain layout.onnx + config.json (id2label).
  static LayoutDetector load(const std::string& model_dir, float conf = 0.45f);
  ~LayoutDetector();
  LayoutDetector(LayoutDetector&&) noexcept;
  LayoutDetector& operator=(LayoutDetector&&) noexcept;

  // Detect on an already-800x800 RGB8 image (row-major, w*h*3). target_w/h scale
  // the boxes to the desired output coordinates (use 800,800 for the model frame
  // or the original page size to map back).
  std::vector<LayoutBox> detect_800(const std::vector<uint8_t>& rgb800, int target_w,
                                    int target_h) const;

  // Detect on an arbitrary-size RGB8 page image: resizes to 800x800 (bicubic),
  // returns boxes in the page's pixel coordinates.
  std::vector<LayoutBox> detect(const std::vector<uint8_t>& rgb, int w, int h) const;

 private:
  LayoutDetector();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

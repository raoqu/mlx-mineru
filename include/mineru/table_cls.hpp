// Copyright (c) mlx-mineru.
// Table-type classifier (PP-LCNet, pipeline backend) via ONNX Runtime:
// a table crop -> wired / wireless. Faithful to MinerU PaddleTableClsModel.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

struct TableClsResult {
  int cls_id;             // 0 = wired_table, 1 = wireless_table
  std::string label;      // "wired_table" / "wireless_table"
  float score;            // max prob
  std::array<float, 2> probs;
};

class TableClassifier {
 public:
  static TableClassifier load(const std::string& onnx_path);
  ~TableClassifier();
  TableClassifier(TableClassifier&&) noexcept;
  TableClassifier& operator=(TableClassifier&&) noexcept;

  // rgb: w*h*3 RGB8 table crop.
  TableClsResult classify(const std::vector<uint8_t>& rgb, int w, int h) const;

 private:
  TableClassifier();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mineru

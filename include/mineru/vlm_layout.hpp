// Copyright (c) mlx-mineru.
// Parser for the Qwen2-VL layout-detection output — faithful port of
// mineru-vl-utils parse_layout_output / _convert_bbox. The model emits, per
// region: <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>TYPE<|ref_end|>
// (<|rotate_*|>)? tail. Coords are integers 0..1000 (normalized to 0..1 here).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace mineru {

struct ContentBlock {
  std::string type;             // normalized block type
  std::array<double, 4> bbox;   // x1,y1,x2,y2 in [0,1]
  std::optional<int> angle;     // 0/90/180/270 or none
  bool merge_prev = false;      // text "txt_contd_tgt" continuation flag
};

// Parse the raw decoded layout output into blocks (invalid/unknown skipped,
// "unknown"->"image", "inline_formula" dropped, table-internal text/equation
// filtered). Mirrors MinerUClient.parse_layout_output.
std::vector<ContentBlock> parse_layout_output(const std::string& output);

// Convert integer bbox (0..1000) to normalized [0,1] with validity checks and
// coordinate ordering; returns nullopt if invalid (out of range / degenerate).
std::optional<std::array<double, 4>> convert_bbox(int x1, int y1, int x2, int y2);

}  // namespace mineru

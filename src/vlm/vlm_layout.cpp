// Copyright (c) mlx-mineru.
#include "mineru/vlm_layout.hpp"

#include <regex>
#include <unordered_set>

namespace mineru {
namespace {

// BLOCK_TYPES from mineru_vl_utils/structs.py.
const std::unordered_set<std::string>& block_types() {
  static const std::unordered_set<std::string> s = {
      "text", "title", "table", "equation", "formula_number", "code", "algorithm",
      "aside_text", "ref_text", "index", "phonetic", "list_item", "table_caption",
      "image_caption", "code_caption", "table_footnote", "image_footnote", "header",
      "footer", "page_number", "page_footnote", "image", "chart", "list",
      "image_block", "equation_block", "unknown"};
  return s;
}

std::optional<int> parse_angle(const std::string& tail) {
  if (tail.find("<|rotate_up|>") != std::string::npos) return 0;
  if (tail.find("<|rotate_right|>") != std::string::npos) return 90;
  if (tail.find("<|rotate_down|>") != std::string::npos) return 180;
  if (tail.find("<|rotate_left|>") != std::string::npos) return 270;
  return std::nullopt;
}

std::string to_lower(std::string s) {
  for (char& c : s) c = (c >= 'A' && c <= 'Z') ? c + 32 : c;
  return s;
}

double bbox_cover_ratio(const std::array<double, 4>& inner, const std::array<double, 4>& outer) {
  double ix0 = std::max(inner[0], outer[0]), iy0 = std::max(inner[1], outer[1]);
  double ix1 = std::min(inner[2], outer[2]), iy1 = std::min(inner[3], outer[3]);
  double iw = ix1 - ix0, ih = iy1 - iy0;
  if (iw <= 0 || ih <= 0) return 0.0;
  double inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1]);
  if (inner_area <= 0) return 0.0;
  return (iw * ih) / inner_area;
}

}  // namespace

std::optional<std::array<double, 4>> convert_bbox(int x1, int y1, int x2, int y2) {
  for (int c : {x1, y1, x2, y2})
    if (c < 0 || c > 1000) return std::nullopt;
  if (x2 < x1) std::swap(x1, x2);
  if (y2 < y1) std::swap(y1, y2);
  if (x1 == x2 || y1 == y2) return std::nullopt;
  return std::array<double, 4>{x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0};
}

std::vector<ContentBlock> parse_layout_output(const std::string& output) {
  // _layout_re with DOTALL: tail uses [\s\S] since ECMAScript '.' excludes newline.
  static const std::regex re(
      R"(<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?([\s\S]*?)(?=<\|box_start\|>|$))");

  std::vector<ContentBlock> blocks;
  auto begin = std::sregex_iterator(output.begin(), output.end(), re);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    auto m = *it;
    auto bbox = convert_bbox(std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3]), std::stoi(m[4]));
    if (!bbox) continue;
    std::string ref_type = to_lower(m[5].str());
    if (ref_type == "unknown") ref_type = "image";
    if (ref_type == "inline_formula") continue;
    if (block_types().find(ref_type) == block_types().end()) continue;
    std::string rotate_tok = m[6].matched ? m[6].str() : "";
    std::string tail = m[7].str();
    std::optional<int> angle = rotate_tok.empty() ? std::nullopt : parse_angle(rotate_tok);
    ContentBlock b;
    b.type = ref_type;
    b.bbox = *bbox;
    b.angle = angle;
    if (ref_type == "text") b.merge_prev = tail.find("txt_contd_tgt") != std::string::npos;
    blocks.push_back(std::move(b));
  }

  // Filter table-internal text/equation/equation_block blocks (covered >= 0.9 by a table).
  std::vector<size_t> table_idx;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (blocks[i].type == "table") table_idx.push_back(i);
  if (table_idx.empty()) return blocks;
  std::unordered_set<size_t> drop;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const std::string& t = blocks[i].type;
    if (t != "text" && t != "equation" && t != "equation_block") continue;
    for (size_t ci : table_idx) {
      if (ci == i) continue;
      if (bbox_cover_ratio(blocks[i].bbox, blocks[ci].bbox) >= 0.9) { drop.insert(i); break; }
    }
  }
  std::vector<ContentBlock> out;
  for (size_t i = 0; i < blocks.size(); ++i)
    if (!drop.count(i)) out.push_back(std::move(blocks[i]));
  return out;
}

}  // namespace mineru

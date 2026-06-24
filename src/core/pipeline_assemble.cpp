// Copyright (c) mlx-mineru.
// Pipeline assembly: layout_dets -> middle_json page_info (text/title path), faithful to
// MinerU MagicModel (__fix_axis / __post_process / __build_page_blocks) +
// model_json_to_middle_json (_post_block_process) + para_split (bbox_fs).
#include "mineru/pipeline_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace mineru {
namespace {
using nlohmann::json;

// PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES (the subset that maps to text-ish blocks; visual
// types image/table/chart/equation are handled in a later slice).
const std::map<std::string, std::string>& labels_map() {
  static const std::map<std::string, std::string> m = {
      {"abstract", "abstract"},        {"aside_text", "aside_text"},
      {"content", "index"},            {"doc_title", "doc_title"},
      {"figure_title", "caption"},     {"footer", "footer"},
      {"footer_image", "footer"},      {"footnote", "page_footnote"},
      {"formula_number", "formula_number"}, {"header", "header"},
      {"header_image", "header"},      {"number", "page_number"},
      {"paragraph_title", "paragraph_title"}, {"reference_content", "ref_text"},
      {"text", "text"},                {"vertical_text", "vertical_text"},
      {"vision_footnote", "footnote"}, {"display_formula", "interline_equation"},
      {"table", "table"},
  };
  return m;
}

// Discarded block types (MagicModel.__build_return_blocks).
bool is_discarded(const std::string& t) {
  return t == "header" || t == "footer" || t == "page_number" || t == "aside_text" ||
         t == "page_footnote";
}

struct Box {
  int x0, y0, x1, y1;
};
Box scale_box(const json& b, double scale) {
  return {(int)(b[0].get<double>() / scale), (int)(b[1].get<double>() / scale),
          (int)(b[2].get<double>() / scale), (int)(b[3].get<double>() / scale)};
}
json box_json(const Box& b) { return json::array({b.x0, b.y0, b.x1, b.y1}); }

// span area ratio inside block (calculate_overlap_area_in_bbox1_area_ratio).
double overlap_in_span_ratio(const Box& s, const Box& blk) {
  int ix0 = std::max(s.x0, blk.x0), iy0 = std::max(s.y0, blk.y0);
  int ix1 = std::min(s.x1, blk.x1), iy1 = std::min(s.y1, blk.y1);
  if (ix1 <= ix0 || iy1 <= iy0) return 0.0;
  double inter = (double)(ix1 - ix0) * (iy1 - iy0);
  double area = (double)(s.x1 - s.x0) * (s.y1 - s.y0);
  return area > 0 ? inter / area : 0.0;
}

// _is_overlaps_y_exceeds_threshold (min-height overlap ratio > thr).
bool overlaps_y(const Box& a, const Box& b, double thr) {
  double overlap = std::max(0, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
  double min_h = std::min(a.y1 - a.y0, b.y1 - b.y0);
  return min_h > 0 ? (overlap / min_h) > thr : false;
}

struct Span {
  Box bbox;
  std::string content;
  double score;
};

// merge_spans_to_line + line_sort_spans_by_left_to_right -> line objects.
json build_lines(std::vector<Span> spans) {
  json lines = json::array();
  if (spans.empty()) return lines;
  std::stable_sort(spans.begin(), spans.end(),
                   [](const Span& a, const Span& b) { return a.bbox.y0 < b.bbox.y0; });
  std::vector<std::vector<Span>> grouped;
  std::vector<Span> cur{spans[0]};
  for (size_t i = 1; i < spans.size(); ++i) {
    if (overlaps_y(spans[i].bbox, cur.back().bbox, 0.6)) {
      cur.push_back(spans[i]);
    } else {
      grouped.push_back(cur);
      cur = {spans[i]};
    }
  }
  grouped.push_back(cur);
  for (auto& line : grouped) {
    std::stable_sort(line.begin(), line.end(),
                     [](const Span& a, const Span& b) { return a.bbox.x0 < b.bbox.x0; });
    int x0 = line[0].bbox.x0, y0 = line[0].bbox.y0, x1 = line[0].bbox.x1, y1 = line[0].bbox.y1;
    json sp = json::array();
    for (const Span& s : line) {
      x0 = std::min(x0, s.bbox.x0); y0 = std::min(y0, s.bbox.y0);
      x1 = std::max(x1, s.bbox.x1); y1 = std::max(y1, s.bbox.y1);
      sp.push_back({{"bbox", box_json(s.bbox)}, {"type", "text"}, {"content", s.content},
                    {"score", s.score}});
    }
    lines.push_back({{"bbox", json::array({x0, y0, x1, y1})}, {"spans", sp}});
  }
  return lines;
}

}  // namespace

json assemble_page_info(const json& model_page, int page_w, int page_h, int page_idx) {
  const json& dets = model_page["layout_dets"];
  double model_w = model_page["page_info"]["width"].get<double>();
  double scale = model_w / page_w;

  // __fix_axis + __post_process: scale boxes, drop tiny; split into spans (ocr_text) and
  // region blocks (reindexed 1..N in array order).
  std::vector<Span> spans;
  struct Region {
    Box bbox; std::string type; double score; int index; std::string latex; std::string html;
  };
  std::vector<Region> regions;
  int next_index = 1;
  for (const json& d : dets) {
    Box b = scale_box(d["bbox"], scale);
    if (b.x1 - b.x0 <= 2 || b.y1 - b.y0 <= 2) continue;
    const std::string label = d.value("label", "");
    if (label == "ocr_text") {
      spans.push_back({b, d.value("text", ""), d.value("score", 1.0)});
      continue;
    }
    auto it = labels_map().find(label);
    if (it == labels_map().end()) continue;
    regions.push_back({b, it->second, d.value("score", 0.0), next_index++, d.value("latex", ""),
                       d.value("html", "")});
  }

  // __build_page_blocks: greedily match unused spans into each block (overlap>0.5), then
  // merge into lines. _post_block_process: title-type conversion.
  std::vector<bool> used(spans.size(), false);
  std::vector<json> blocks;  // preproc blocks (no bbox_fs yet)
  for (const Region& r : regions) {
    // Interline equation: one latex span, no OCR-text matching (__build_page_blocks visual).
    if (r.type == "interline_equation") {
      json span = {{"bbox", box_json(r.bbox)}, {"type", "interline_equation"},
                   {"content", r.latex}};
      json line = {{"bbox", box_json(r.bbox)}, {"spans", json::array({span})}};
      blocks.push_back({{"score", r.score}, {"bbox", box_json(r.bbox)}, {"index", r.index},
                        {"type", "interline_equation"}, {"lines", json::array({line})}});
      continue;
    }
    // Table: two-layer block {type:table, blocks:[table_body(html span)]}. NOTE: caption/
    // footnote association (find_best_visual_parent) is a documented follow-up, so captions
    // (figure_title) and footnotes (vision_footnote) stay as their own blocks for now.
    if (r.type == "table") {
      json span = {{"bbox", box_json(r.bbox)}, {"type", "table"}, {"html", r.html}};
      json line = {{"bbox", box_json(r.bbox)}, {"spans", json::array({span})}};
      json body = {{"score", r.score}, {"bbox", box_json(r.bbox)}, {"index", r.index},
                   {"type", "table_body"}, {"lines", json::array({line})}};
      blocks.push_back({{"type", "table"}, {"bbox", box_json(r.bbox)}, {"index", r.index},
                        {"score", r.score}, {"blocks", json::array({body})}});
      continue;
    }
    std::vector<Span> blk_spans;
    for (size_t i = 0; i < spans.size(); ++i) {
      if (used[i]) continue;
      if (overlap_in_span_ratio(spans[i].bbox, r.bbox) > 0.5) {
        blk_spans.push_back(spans[i]);
        used[i] = true;
      }
    }
    std::string type = r.type;
    int level = 0;
    bool has_level = false;
    if (type == "doc_title") { type = "title"; level = 1; has_level = true; }
    else if (type == "paragraph_title") { type = "title"; level = 2; has_level = true; }
    else if (type == "vertical_text") { type = "text"; }

    json block = {{"score", r.score}, {"bbox", box_json(r.bbox)}, {"index", r.index},
                  {"type", type}, {"lines", build_lines(blk_spans)}};
    if (has_level) block["level"] = level;
    blocks.push_back(block);
  }

  std::stable_sort(blocks.begin(), blocks.end(),
                   [](const json& a, const json& b) { return a["index"] < b["index"]; });

  // __build_return_blocks: split preproc vs discarded.
  json preproc = json::array(), discarded = json::array();
  for (const json& b : blocks) {
    if (is_discarded(b["type"].get<std::string>())) discarded.push_back(b);
    else preproc.push_back(b);
  }

  // para_split (text path): para_blocks = preproc + bbox_fs on text blocks.
  json para = json::array();
  for (json b : preproc) {
    if (b["type"] == "text" && !b["lines"].empty()) {
      const json& lines = b["lines"];
      int x0 = lines[0]["bbox"][0], y0 = lines[0]["bbox"][1];
      int x1 = lines[0]["bbox"][2], y1 = lines[0]["bbox"][3];
      for (const json& ln : lines) {
        x0 = std::min(x0, ln["bbox"][0].get<int>()); y0 = std::min(y0, ln["bbox"][1].get<int>());
        x1 = std::max(x1, ln["bbox"][2].get<int>()); y1 = std::max(y1, ln["bbox"][3].get<int>());
      }
      b["bbox_fs"] = json::array({x0, y0, x1, y1});
    }
    para.push_back(b);
  }

  return {{"preproc_blocks", preproc}, {"page_idx", page_idx},
          {"page_size", json::array({page_w, page_h})},
          {"discarded_blocks", discarded}, {"para_blocks", para}};
}

}  // namespace mineru

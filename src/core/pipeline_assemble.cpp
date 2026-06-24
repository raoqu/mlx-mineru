// Copyright (c) mlx-mineru.
// Pipeline assembly: layout_dets -> middle_json page_info (text/title path), faithful to
// MinerU MagicModel (__fix_axis / __post_process / __build_page_blocks) +
// model_json_to_middle_json (_post_block_process) + para_split (bbox_fs).
#include "mineru/pipeline_assemble.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
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
      {"table", "table"},             {"image", "image"},
      {"chart", "chart"},
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

// ---- visual block classify (caption/footnote -> table/image/chart) -----------
using Bbox4 = std::array<double, 4>;
Bbox4 block_bbox(const json& b) {
  return {b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]};
}
bool is_main_type(const std::string& t) { return t == "table" || t == "image" || t == "chart"; }
bool is_child_type(const std::string& t) { return t == "caption" || t == "footnote"; }

bool boxes_overlap(const Bbox4& a, const Bbox4& b) {
  return !(a[2] <= b[0] || a[0] >= b[2] || a[3] <= b[1] || a[1] >= b[3]);
}
void relpos(const Bbox4& a, const Bbox4& b, bool& left, bool& right, bool& bottom, bool& top) {
  left = b[2] < a[0]; right = a[2] < b[0]; bottom = b[3] < a[1]; top = a[3] < b[1];
}
double bbox_distance(const Bbox4& a, const Bbox4& b) {
  auto d = [](double px, double py, double qx, double qy) {
    return std::hypot(px - qx, py - qy);
  };
  bool l, r, bo, t;
  relpos(a, b, l, r, bo, t);
  if (t && l) return d(a[0], a[3], b[2], b[1]);
  if (l && bo) return d(a[0], a[1], b[2], b[3]);
  if (bo && r) return d(a[2], a[1], b[0], b[3]);
  if (r && t) return d(a[2], a[3], b[0], b[1]);
  if (l) return a[0] - b[2];
  if (r) return b[0] - a[2];
  if (bo) return a[1] - b[3];
  if (t) return b[1] - a[3];
  return 0.0;
}
double bbox_center_dist(const Bbox4& a, const Bbox4& b) {
  return std::hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2);
}
// vertical_gap_between_blocks: {top,bottom} of the gap if separated, else invalid (.first<0).
std::array<double, 2> vgap(const Bbox4& a, const Bbox4& b) {
  if (a[3] <= b[1]) return {a[3], b[1]};
  if (b[3] <= a[1]) return {b[3], a[1]};
  return {-1, -1};
}

// is_visual_neighbor: child & main are adjacent in reading order with only allowed types
// (or visually-outside-the-gap blocks) between them.
bool is_visual_neighbor(int ci, int mi, const std::vector<json>& ord) {
  const json& child = ord[ci];
  const json& main = ord[mi];
  std::string ct = child["type"];
  if (ct == "footnote" && child["index"].get<int>() < main["index"].get<int>()) return false;
  Bbox4 cb = block_bbox(child), mb = block_bbox(main);
  auto gap = vgap(cb, mb);
  int a = std::min(ci, mi) + 1, z = std::max(ci, mi);
  for (int p = a; p < z; ++p) {
    std::string bt = ord[p]["type"];
    bool allowed = (ct == "caption") ? (bt == "caption")
                                     : (bt == "caption" || bt == "footnote");
    if (allowed) continue;
    // is_block_outside_visual_gap: between block not overlapping child/main and not in gap.
    Bbox4 bb = block_bbox(ord[p]);
    bool outside = false;
    if (gap[0] >= 0 && !(boxes_overlap(bb, cb) || boxes_overlap(bb, mb)) &&
        !(bb[1] < gap[1] && bb[3] > gap[0]))
      outside = true;
    if (outside) continue;
    return false;
  }
  return true;
}
int eff_index_diff(int ci, int mi, const std::vector<json>& ord) {
  std::string ct = ord[ci]["type"];
  int a = std::min(ci, mi), z = std::max(ci, mi), skipped = 0;
  for (int p = a + 1; p < z; ++p)
    if (ord[p]["type"] == ct) ++skipped;
  return z - a - skipped;
}
// find_best_visual_parent: returns the index into `ord` of the best main block, or -1.
int find_best_parent(int ci, const std::vector<int>& mains, const std::vector<json>& ord) {
  std::vector<int> cand;
  for (int mi : mains)
    if (is_visual_neighbor(ci, mi, ord)) cand.push_back(mi);
  if (cand.empty()) return -1;
  int best_diff = 1 << 30;
  for (int mi : cand) best_diff = std::min(best_diff, eff_index_diff(ci, mi, ord));
  std::vector<int> closest;
  for (int mi : cand)
    if (eff_index_diff(ci, mi, ord) == best_diff) closest.push_back(mi);
  if (closest.size() == 1) return closest[0];
  Bbox4 cb = block_bbox(ord[ci]);
  std::vector<double> edges;
  for (int mi : closest) edges.push_back(bbox_distance(cb, block_bbox(ord[mi])));
  double emin = *std::min_element(edges.begin(), edges.end());
  double emax = *std::max_element(edges.begin(), edges.end());
  if (emax - emin > 2) {  // pick min edge distance, tie-break by index
    int best = closest[0];
    double bd = 1e300;
    for (int mi : closest) {
      double d = bbox_distance(cb, block_bbox(ord[mi]));
      if (d < bd || (d == bd && ord[mi]["index"] < ord[best]["index"])) { bd = d; best = mi; }
    }
    return best;
  }
  std::string ct = ord[ci]["type"];
  if (ct == "caption") {  // tables: prefer the later table when equidistant
    bool all_table = true;
    for (int mi : closest) if (ord[mi]["type"] != "table") all_table = false;
    if (all_table) {
      int best = closest[0];
      for (int mi : closest) if (ord[mi]["index"] > ord[best]["index"]) best = mi;
      return best;
    }
  }
  if (ct == "footnote") {  // prefer the earlier main
    int best = closest[0];
    for (int mi : closest) if (ord[mi]["index"] < ord[best]["index"]) best = mi;
    return best;
  }
  int best = closest[0];
  double bd = 1e300;
  for (int mi : closest) {
    double d = bbox_center_dist(cb, block_bbox(ord[mi]));
    if (d < bd || (d == bd && ord[mi]["index"] < ord[best]["index"])) { bd = d; best = mi; }
  }
  return best;
}

// __classify_visual_blocks: wrap each table/image/chart main block as a two-layer block with
// its associated caption/footnote children (figure_title->caption, vision_footnote->footnote).
void classify_visual_blocks(std::vector<json>& blocks) {
  std::sort(blocks.begin(), blocks.end(),
            [](const json& a, const json& b) { return a["index"] < b["index"]; });
  std::vector<int> mains, children;
  for (int i = 0; i < (int)blocks.size(); ++i) {
    std::string t = blocks[i]["type"];
    if (is_main_type(t)) mains.push_back(i);
    else if (is_child_type(t)) children.push_back(i);
  }
  if (mains.empty()) return;
  std::map<int, int> child_parent;  // child idx -> main idx (in blocks), -1 if none
  for (int ci : children) child_parent[ci] = find_best_parent(ci, mains, blocks);

  // body/caption/footnote type names per main type.
  auto body_type = [](const std::string& m) { return m + "_body"; };
  auto cap_type = [](const std::string& m) { return m + "_caption"; };
  auto foot_type = [](const std::string& m) { return m + "_footnote"; };
  std::map<int, std::vector<int>> caps, foots;
  for (int ci : children) {
    int pi = child_parent[ci];
    if (pi < 0) continue;
    (blocks[ci]["type"] == "caption" ? caps : foots)[pi].push_back(ci);
  }

  std::vector<json> out;
  for (int i = 0; i < (int)blocks.size(); ++i) {
    std::string t = blocks[i]["type"];
    if (is_child_type(t)) {
      if (child_parent[i] < 0) { blocks[i]["type"] = "text"; out.push_back(blocks[i]); }
      continue;  // associated children are emitted under their parent
    }
    if (!is_main_type(t)) { out.push_back(blocks[i]); continue; }
    json body = blocks[i];
    body["type"] = body_type(t);
    json children_blocks = json::array();
    children_blocks.push_back(body);
    for (int ci : caps[i]) { json c = blocks[ci]; c["type"] = cap_type(t); children_blocks.push_back(c); }
    for (int ci : foots[i]) { json c = blocks[ci]; c["type"] = foot_type(t); children_blocks.push_back(c); }
    std::sort(children_blocks.begin(), children_blocks.end(),
              [](const json& a, const json& b) { return a["index"] < b["index"]; });
    out.push_back({{"type", t}, {"bbox", blocks[i]["bbox"]}, {"blocks", children_blocks},
                   {"index", blocks[i]["index"]}, {"score", blocks[i].value("score", 0.0)}});
  }
  blocks = std::move(out);
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
    // Table: flat block carrying the html span; classify_visual_blocks wraps it into a
    // two-layer block and attaches associated caption/footnote children.
    if (r.type == "table") {
      json span = {{"bbox", box_json(r.bbox)}, {"type", "table"}, {"html", r.html}};
      json line = {{"bbox", box_json(r.bbox)}, {"spans", json::array({span})}};
      blocks.push_back({{"score", r.score}, {"bbox", box_json(r.bbox)}, {"index", r.index},
                        {"type", "table"}, {"lines", json::array({line})}});
      continue;
    }
    // Image / chart: flat visual block with an image span (image_path + content filled later
    // by the VLM in hybrid mode); classify_visual_blocks wraps it + attaches caption/footnote.
    if (r.type == "image" || r.type == "chart") {
      json span = {{"bbox", box_json(r.bbox)}, {"type", r.type}, {"image_path", ""},
                   {"content", ""}};
      json line = {{"bbox", box_json(r.bbox)}, {"spans", json::array({span})}};
      blocks.push_back({{"score", r.score}, {"bbox", box_json(r.bbox)}, {"index", r.index},
                        {"type", r.type}, {"lines", json::array({line})}});
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

  // __classify_visual_blocks: wrap table/image/chart with associated caption/footnote.
  classify_visual_blocks(blocks);

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

// ---- formula_number -> \tag{N} association (optimize_formula_number_blocks) ----
namespace {
std::string strip_ws(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}
// extract_formula_number_text: concat the block's text-span contents.
std::string fn_text(const json& block) {
  std::string t;
  for (const auto& ln : block.value("lines", json::array()))
    for (const auto& sp : ln.value("spans", json::array()))
      if (sp.value("type", "") == "text") t += sp.value("content", "");
  return strip_ws(t);
}
// normalize_formula_tag_content: strip whitespace + a single outer pair of (full/half)
// parentheses around the number.
std::string normalize_tag(std::string s) {
  s = strip_ws(s);
  auto starts = [&](const char* p) { return s.rfind(p, 0) == 0; };
  auto ends = [&](const char* p) {
    size_t n = std::string(p).size();
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
  };
  if (starts("(")) s = strip_ws(s.substr(1));
  else if (starts("\xEF\xBC\x88")) s = strip_ws(s.substr(3));  // U+FF08 （
  if (ends(")")) s = strip_ws(s.substr(0, s.size() - 1));
  else if (ends("\xEF\xBC\x89")) s = strip_ws(s.substr(0, s.size() - 3));  // U+FF09 ）
  return s;
}
// append_formula_number_tag: write \tag{N} into the equation block's interline span.
bool append_tag(json& eq_block, const json& fn_block) {
  for (auto& ln : eq_block["lines"])
    for (auto& sp : ln["spans"])
      if (sp.value("type", "") == "interline_equation") {
        std::string content = sp.value("content", "");
        if (strip_ws(content).empty()) return false;
        sp["content"] = content + "\\tag{" + normalize_tag(fn_text(fn_block)) + "}";
        return true;
      }
  return false;
}
bool is_eq(const json& b) { return b.value("type", "") == "interline_equation"; }
bool is_fn(const json& b) { return b.value("type", "") == "formula_number"; }
}  // namespace

// ---- digital-PDF text fill (txt_spans_extract / fill_char_in_spans) ----------
namespace {
std::string cp_to_utf8(unsigned int cp) {
  std::string s;
  if (cp < 0x80) s += (char)cp;
  else if (cp < 0x800) { s += (char)(0xC0 | cp >> 6); s += (char)(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) {
    s += (char)(0xE0 | cp >> 12); s += (char)(0x80 | ((cp >> 6) & 0x3F));
    s += (char)(0x80 | (cp & 0x3F));
  } else {
    s += (char)(0xF0 | cp >> 18); s += (char)(0x80 | ((cp >> 12) & 0x3F));
    s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F));
  }
  return s;
}

// LINE_STOP_FLAG / LINE_START_FLAG as codepoints (the ASCII + common CJK punctuation that
// get the relaxed edge test in calculate_char_in_span).
bool is_stop_flag(unsigned int c) {
  static const std::set<unsigned int> s = {'.', '!', '?', 0x3002, 0xFF01, 0xFF1F, ')', 0xFF09,
      '"', 0x201D, ':', 0xFF1A, ';', 0xFF1B, ']', 0x3011, '}', '>', 0x300B, 0x3001, ',',
      0xFF0C, '-', 0x2014, 0x2013};
  return s.count(c) > 0;
}
bool is_start_flag(unsigned int c) {
  static const std::set<unsigned int> s = {'(', 0xFF08, '"', 0x201C, 0x3010, '{', 0x300A, '<',
      0x300C, 0x300E, '['};
  return s.count(c) > 0;
}

// calculate_char_in_span: center-in-span + vertical-axis alignment, with relaxed edge tests
// for line stop/start punctuation.
bool char_in_span(const PageChar& ch, const std::array<double, 4>& sb) {
  double ccx = (ch.x0 + ch.x1) / 2, ccy = (ch.y0 + ch.y1) / 2;
  double scy = (sb[1] + sb[3]) / 2, sh = sb[3] - sb[1];
  const double R = 0.33;
  if (sb[0] < ccx && ccx < sb[2] && sb[1] < ccy && ccy < sb[3] &&
      std::abs(ccy - scy) < sh * R)
    return true;
  if (is_stop_flag(ch.cp))
    return (sb[2] - sh) < ch.x0 && ch.x0 < sb[2] && ccx > sb[0] && sb[1] < ccy && ccy < sb[3] &&
           std::abs(ccy - scy) < sh * R;
  if (is_start_flag(ch.cp))
    return sb[0] < ch.x1 && ch.x1 < (sb[0] + sh) && ccx < sb[2] && sb[1] < ccy && ccy < sb[3] &&
           std::abs(ccy - scy) < sh * R;
  return false;
}

// chars_to_content: sort by original index, drop control line-breaks, concat with a space
// inserted where the inter-char gap exceeds 0.25 * median char width.
std::string chars_to_content(std::vector<const PageChar*>& cs) {
  std::sort(cs.begin(), cs.end(), [](const PageChar* a, const PageChar* b) { return a->idx < b->idx; });
  std::vector<const PageChar*> kept;
  for (auto* c : cs)
    if (c->cp != 10 && c->cp != 13) kept.push_back(c);  // drop \n \r
  if (kept.empty()) return "";
  std::vector<double> widths;
  for (auto* c : kept) widths.push_back(c->x1 - c->x0);
  std::sort(widths.begin(), widths.end());
  double mw = widths[widths.size() / 2];
  std::string out;
  for (size_t i = 0; i < kept.size(); ++i) {
    out += cp_to_utf8(kept[i]->cp);
    if (i + 1 < kept.size() && kept[i + 1]->x0 - kept[i]->x1 > mw * 0.25 && kept[i]->cp != ' ' &&
        kept[i + 1]->cp != ' ')
      out += ' ';
  }
  return out;
}

// Collect text-span pointers from a block list, descending into nested "blocks" (two-layer
// table/image/chart blocks).
struct SpanRef { json* sp; std::array<double, 4> bbox; std::vector<const PageChar*> cs; };
void collect_text_spans(json& blocks, std::vector<SpanRef>& spans) {
  for (auto& b : blocks) {
    if (b.contains("lines"))
      for (auto& ln : b["lines"])
        for (auto& sp : ln["spans"])
          if (sp.value("type", "") == "text")
            spans.push_back({&sp, {sp["bbox"][0], sp["bbox"][1], sp["bbox"][2], sp["bbox"][3]}, {}});
    if (b.contains("blocks")) collect_text_spans(b["blocks"], spans);
  }
}

// Fill one block list's text spans from the page chars.
int fill_one(json& blocks, const std::vector<PageChar>& chars) {
  std::vector<SpanRef> spans;
  collect_text_spans(blocks, spans);
  std::sort(spans.begin(), spans.end(),
            [](const SpanRef& a, const SpanRef& b) { return a.bbox[1] < b.bbox[1]; });
  // Assign each char to the first span (top-to-bottom) it falls in.
  for (const PageChar& ch : chars) {
    double ccx = (ch.x0 + ch.x1) / 2;
    for (auto& s : spans) {
      if (!is_stop_flag(ch.cp) && !is_start_flag(ch.cp) && !(s.bbox[0] < ccx && ccx < s.bbox[2]))
        continue;
      if (char_in_span(ch, s.bbox)) { s.cs.push_back(&ch); break; }
    }
  }
  int empty = 0;
  for (auto& s : spans) {
    std::string content = chars_to_content(s.cs);
    (*s.sp)["content"] = content;
    (*s.sp)["score"] = content.empty() ? 0.0 : 1.0;
    if (content.empty()) ++empty;
  }
  return empty;
}
}  // namespace

int fill_chars_in_page(json& page_info, const std::vector<PageChar>& chars) {
  int empty = 0;
  for (const char* key : {"preproc_blocks", "discarded_blocks", "para_blocks"})
    if (page_info.contains(key)) empty += fill_one(page_info[key], chars);
  return empty;
}

void optimize_formula_numbers(json& blocks) {
  json out = json::array();
  int n = (int)blocks.size();
  for (int i = 0; i < n; ++i) {
    if (!is_fn(blocks[i])) { out.push_back(blocks[i]); continue; }
    // prev block is an equation -> append tag to it (already in `out`), drop this.
    if (i > 0 && !out.empty() && is_eq(out.back())) {
      if (append_tag(out.back(), blocks[i])) continue;
      blocks[i]["type"] = "text";
      out.push_back(blocks[i]);
      continue;
    }
    // next is an equation (and the one after isn't another formula_number) -> tag it.
    if (i + 1 < n && is_eq(blocks[i + 1]) && (i + 2 >= n || !is_fn(blocks[i + 2]))) {
      if (append_tag(blocks[i + 1], blocks[i])) continue;  // tagged when pushed next iter
      blocks[i]["type"] = "text";
      out.push_back(blocks[i]);
      continue;
    }
    blocks[i]["type"] = "text";  // unmatched -> downgrade
    out.push_back(blocks[i]);
  }
  blocks = std::move(out);
}

}  // namespace mineru

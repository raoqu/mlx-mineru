// Copyright (c) mlx-mineru.
// Faithful port of mineru/backend/vlm/vlm_middle_json_mkcontent.py.
// Operates directly on nlohmann::json (like the Python dict code) to guarantee
// field-level parity. Line references in comments point to the Python source.
#include "mineru/mkcontent.hpp"

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "mineru/enums.hpp"
#include "text_utils.hpp"

namespace mineru {
namespace {

using nlohmann::json;

// content_type_v2 string constants (mirrors ContentTypeV2 in enum_class.py).
namespace content_type_v2 {
inline constexpr const char* kCode = "code";
inline constexpr const char* kAlgorithm = "algorithm";
inline constexpr const char* kEquationInterline = "equation_interline";
inline constexpr const char* kImage = "image";
inline constexpr const char* kTable = "table";
inline constexpr const char* kChart = "chart";
inline constexpr const char* kTableSimple = "simple_table";
inline constexpr const char* kTableComplex = "complex_table";
inline constexpr const char* kList = "list";
inline constexpr const char* kListText = "text_list";
inline constexpr const char* kListRef = "reference_list";
inline constexpr const char* kTitle = "title";
inline constexpr const char* kParagraph = "paragraph";
inline constexpr const char* kSpanText = "text";
inline constexpr const char* kSpanEquationInline = "equation_inline";
inline constexpr const char* kSpanPhonetic = "phonetic";
inline constexpr const char* kPageHeader = "page_header";
inline constexpr const char* kPageFooter = "page_footer";
inline constexpr const char* kPageNumber = "page_number";
inline constexpr const char* kPageAsideText = "page_aside_text";
inline constexpr const char* kPageFootnote = "page_footnote";
}  // namespace content_type_v2

namespace bt = block_type;
namespace ct = content_type;
namespace ctv2 = content_type_v2;

// Per-call rendering context (delimiters + feature flags), avoids globals.
struct Ctx {
  LatexDelimiters d;
  bool formula_enable = true;
  bool table_enable = true;
};

// ---- small string helpers ---------------------------------------------------
std::string strip(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}
bool is_blank(const std::string& s) { return strip(s).empty(); }

std::string get_str(const json& j, const char* key, const std::string& def = "") {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return def;
  if (it->is_string()) return it->get<std::string>();
  return def;
}
bool has_nonempty_str(const json& j, const char* key) {
  auto it = j.find(key);
  return it != j.end() && it->is_string() && !it->get<std::string>().empty();
}

std::string remove_last_codepoint(const std::string& s) {
  auto cps = text::utf8_decode(s);
  if (cps.empty()) return s;
  cps.pop_back();
  return text::utf8_encode(cps);
}

// ---- table HTML normalization (lines 33-62) ---------------------------------
std::string prefix_table_img_src(const std::string& html, const std::string& bucket) {
  if (html.empty() || bucket.empty()) return html;
  static const std::regex re("src=\"(?!data:)([^\"]+)\"");
  return std::regex_replace(html, re, "src=\"" + bucket + "/$1\"");
}

std::string html_unescape(const std::string& s) {
  // Minimal html.unescape: named basics + numeric refs.
  std::string out;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '&') {
      size_t semi = s.find(';', i);
      if (semi != std::string::npos && semi - i <= 10) {
        std::string ent = s.substr(i + 1, semi - i - 1);
        std::string rep;
        if (ent == "amp") rep = "&";
        else if (ent == "lt") rep = "<";
        else if (ent == "gt") rep = ">";
        else if (ent == "quot") rep = "\"";
        else if (ent == "#39" || ent == "apos") rep = "'";
        else if (ent == "nbsp") rep = "\xC2\xA0";
        else if (!ent.empty() && ent[0] == '#') {
          long cp = 0;
          try {
            cp = (ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X'))
                     ? std::stol(ent.substr(2), nullptr, 16)
                     : std::stol(ent.substr(1));
          } catch (...) { cp = -1; }
          if (cp >= 0) { std::string t; text::utf8_append(t, (char32_t)cp); rep = t; }
        }
        if (!rep.empty() || ent == "amp" || ent == "lt" || ent == "gt") {
          out += rep;
          i = semi + 1;
          continue;
        }
      }
    }
    out.push_back(s[i]);
    ++i;
  }
  return out;
}

std::string replace_eq_tags_in_table_html(const std::string& html, const Ctx& c) {
  if (html.empty()) return html;
  static const std::regex re("<eq>([\\s\\S]*?)</eq>");
  std::string out;
  auto begin = std::sregex_iterator(html.begin(), html.end(), re);
  auto end = std::sregex_iterator();
  size_t last = 0;
  for (auto it = begin; it != end; ++it) {
    auto m = *it;
    out += html.substr(last, m.position() - last);
    out += " " + c.d.inline_left + html_unescape(m[1].str()) + c.d.inline_right + " ";
    last = m.position() + m.length();
  }
  out += html.substr(last);
  return out;
}

std::string format_embedded_html(const std::string& html, const std::string& bucket, const Ctx& c) {
  return replace_eq_tags_in_table_html(prefix_table_img_src(html, bucket), c);
}

// ---- misc helpers (lines 65-143) --------------------------------------------
std::string normalize_text_content(const json& span) {
  if (!span.contains("content") || span["content"].is_null()) return "";
  return text::full_to_half_exclude_marks(span["content"].get<std::string>());
}

std::string build_media_path(const std::string& bucket, const std::string& image_path) {
  if (image_path.empty()) return "";
  if (bucket.empty()) return image_path;
  return bucket + "/" + image_path;
}

int get_title_level(const json& block) {  // lines 935-941
  int level = block.contains("level") && block["level"].is_number() ? block["level"].get<int>() : 1;
  if (level > 4) level = 4;
  else if (level < 1) level = 0;
  return level;
}

// _has_following_joinable_span (lines 72-85)
bool has_following_joinable_span(const json& para_block, size_t line_idx, size_t span_idx) {
  const json& lines = para_block["lines"];
  for (size_t nl = line_idx; nl < lines.size(); ++nl) {
    const json& next_line = lines[nl];
    size_t start = (nl == line_idx) ? span_idx + 1 : 0;
    json spans = next_line.value("spans", json::array());
    for (size_t s = start; s < spans.size(); ++s) {
      const json& nsp = spans[s];
      std::string t = nsp.value("type", "");
      if (t == ct::kText) {
        if (!strip(normalize_text_content(nsp)).empty()) return true;
      } else if (t == ct::kInlineEquation) {
        if (!strip(get_str(nsp, "content")).empty()) return true;
      }
    }
  }
  return false;
}

// ---- merge_para_with_text (lines 275-358) -----------------------------------
// `formula_enable` defaults to true: MinerU only threads the user flag through
// the markdown TEXT-family and LIST-item paths; all other callers use True.
std::string merge_para_with_text(const json& para_block, const Ctx& c,
                                 bool escape_text_block_prefix = true,
                                 bool formula_enable = true) {
  std::string block_text;
  for (const auto& line : para_block["lines"])
    for (const auto& span : line["spans"])
      if (span["type"] == ct::kText) block_text += normalize_text_content(span);
  std::string block_lang = text::detect_lang(block_text);
  bool escape_markdown_text = para_block.value("type", "") != bt::kCodeBody;

  std::string para_text;
  const json& lines = para_block["lines"];
  for (size_t i = 0; i < lines.size(); ++i) {
    const json& line = lines[i];
    const json& spans = line["spans"];
    for (size_t j = 0; j < spans.size(); ++j) {
      const json& span = spans[j];
      std::string span_type = span["type"];
      std::string content;
      if (span_type == ct::kText) {
        content = normalize_text_content(span);
        if (escape_markdown_text) content = text::escape_conservative_markdown_text(content);
      } else if (span_type == ct::kInlineEquation) {
        content = c.d.inline_left + get_str(span, "content") + c.d.inline_right;
      } else if (span_type == ct::kInterlineEquation) {
        if (formula_enable) {
          content = "\n" + c.d.display_left + "\n" + get_str(span, "content") + "\n" +
                    c.d.display_right + "\n";
        } else if (has_nonempty_str(span, "image_path")) {
          content = "![](" + /*img_bucket folded into caller via prefix*/ get_str(span, "image_path") + ")";
        }
      }
      content = strip(content);
      if (content.empty()) continue;

      if (span_type == ct::kInterlineEquation) {
        para_text += content;
        continue;
      }
      bool is_last_span = (j == spans.size() - 1);
      bool following = has_following_joinable_span(para_block, i, j);
      if (text::is_cjk_lang(block_lang)) {
        if (following && (!is_last_span || span_type == ct::kInlineEquation))
          para_text += content + " ";
        else
          para_text += content;
      } else {
        if (span_type == ct::kText || span_type == ct::kInlineEquation) {
          if (is_last_span && span_type == ct::kText && text::is_hyphen_at_line_end(content)) {
            bool next_lower = false;
            if (i + 1 < lines.size()) {
              const json& nspans = lines[i + 1].value("spans", json::array());
              if (!nspans.empty() && nspans[0].value("type", "") == ct::kText) {
                std::string nc = get_str(nspans[0], "content");
                if (!nc.empty() && nc[0] >= 'a' && nc[0] <= 'z') next_lower = true;
              }
            }
            para_text += next_lower ? remove_last_codepoint(content) : content;
          } else {
            para_text += following ? content + " " : content;
          }
        }
      }
    }
  }
  if (escape_text_block_prefix && para_block.value("type", "") == bt::kText)
    para_text = text::escape_text_block_markdown_prefix(para_text);
  return para_text;
}

// Variant of merge_para_with_text that resolves interline-equation image refs
// against the bucket (the Python code captures img_buket_path in closure).
std::string merge_para_with_text_img(const json& para_block, const Ctx& c,
                                     const std::string& bucket, bool formula_enable,
                                     bool escape_text_block_prefix = true) {
  // Same as merge_para_with_text but for the (rare) formula-disabled image fallback
  // path we must prefix the bucket. We re-run the logic via a patched copy.
  if (formula_enable) return merge_para_with_text(para_block, c, escape_text_block_prefix, true);
  json patched = para_block;
  for (auto& line : patched["lines"])
    for (auto& span : line["spans"])
      if (span.value("type", "") == ct::kInterlineEquation && span.contains("image_path"))
        span["image_path"] = bucket.empty() ? get_str(span, "image_path")
                                            : bucket + "/" + get_str(span, "image_path");
  return merge_para_with_text(patched, c, escape_text_block_prefix, false);
}

// ---- code block rendering (lines 146-159) -----------------------------------
std::string render_code_block_markdown(const json& block, const json& para_block, const Ctx& c) {
  std::string sub_type = get_str(para_block, "sub_type");
  if (sub_type == bt::kAlgorithm) {
    return text::render_algorithm_html_from_lines(block.value("lines", json::array()),
                                                  c.d.inline_left, c.d.inline_right);
  } else if (sub_type == bt::kCode) {
    std::string code_text = merge_para_with_text(block, c);
    std::string guess = get_str(para_block, "guess_lang", "txt");
    return "```" + guess + "\n" + code_text + "\n```";
  }
  throw std::runtime_error("Unknown code block sub_type: " + sub_type);
}

// ---- visual block rendering (lines 102-272) ---------------------------------
std::string build_visual_details_block(const std::string& content, const std::string& span_type,
                                       const std::string& summary_override) {
  if (strip(content).empty()) return "";
  std::string summary = !summary_override.empty()
                            ? summary_override
                            : (span_type == ct::kChart ? "chart content" : "image content");
  return "<details>\n<summary>" + summary + "</summary>\n\n" + content + "\n</details>";
}

struct Seg { std::string text; std::string kind; };  // kind: markdown_line | html_block

std::vector<Seg> build_visual_body_segments(const std::string& image_path,
                                            const std::string& content, const std::string& bucket,
                                            const std::string& span_type,
                                            const std::string& summary_override) {
  std::vector<Seg> segs;
  std::string media = build_media_path(bucket, image_path);
  if (!media.empty()) segs.push_back({"![](" + media + ")", "markdown_line"});
  std::string details = build_visual_details_block(content, span_type, summary_override);
  if (!details.empty()) segs.push_back({details, "html_block"});
  return segs;
}

std::vector<const json*> blocks_in_index_order(const json& blocks) {  // lines 136-143
  std::vector<std::pair<size_t, const json*>> items;
  for (size_t i = 0; i < blocks.size(); ++i) items.push_back({i, &blocks[i]});
  std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    double ia = a.second->contains("index") && (*a.second)["index"].is_number()
                    ? (*a.second)["index"].template get<double>()
                    : std::numeric_limits<double>::infinity();
    double ib = b.second->contains("index") && (*b.second)["index"].is_number()
                    ? (*b.second)["index"].template get<double>()
                    : std::numeric_limits<double>::infinity();
    if (ia != ib) return ia < ib;
    return a.first < b.first;
  });
  std::vector<const json*> out;
  for (auto& it : items) out.push_back(it.second);
  return out;
}

std::vector<Seg> render_visual_block_segments(const json& block, const json& para_block,
                                              const std::string& bucket, const Ctx& c) {
  std::string block_type = block["type"];
  static const std::vector<std::string> caption_types = {
      bt::kImageCaption, bt::kImageFootnote, bt::kTableCaption, bt::kTableFootnote,
      bt::kCodeCaption,  bt::kCodeFootnote,  bt::kChartCaption, bt::kChartFootnote};
  if (std::find(caption_types.begin(), caption_types.end(), block_type) != caption_types.end()) {
    std::string t = merge_para_with_text(block, c);
    if (!is_blank(t)) return {{t, "markdown_line"}};
    return {};
  }
  if (block_type == bt::kImageBody || block_type == bt::kChartBody) {
    bool is_chart = (block_type == bt::kChartBody);
    std::string want = is_chart ? ct::kChart : ct::kImage;
    std::vector<Seg> segs;
    for (const auto& line : block.value("lines", json::array()))
      for (const auto& span : line.value("spans", json::array()))
        if (span.value("type", "") == want) {
          auto s = build_visual_body_segments(get_str(span, "image_path"), get_str(span, "content"),
                                              bucket, want, get_str(para_block, "sub_type"));
          segs.insert(segs.end(), s.begin(), s.end());
        }
    return segs;
  }
  if (block_type == bt::kTableBody) {
    std::vector<Seg> segs;
    for (const auto& line : block.value("lines", json::array()))
      for (const auto& span : line.value("spans", json::array())) {
        if (span.value("type", "") != ct::kTable) continue;
        if (c.table_enable && has_nonempty_str(span, "html"))
          segs.push_back({format_embedded_html(get_str(span, "html"), bucket, c), "html_block"});
        else if (has_nonempty_str(span, "image_path"))
          segs.push_back(
              {"![](" + build_media_path(bucket, get_str(span, "image_path")) + ")", "markdown_line"});
      }
    return segs;
  }
  if (block_type == bt::kCodeBody) {
    std::string t = render_code_block_markdown(block, para_block, c);
    if (!is_blank(t)) {
      std::string kind = (get_str(para_block, "sub_type") == bt::kAlgorithm) ? "html_block" : "markdown_line";
      return {{t, kind}};
    }
    return {};
  }
  return {};
}

std::string visual_block_separator(const std::string& prev, const std::string& cur) {
  if (prev == "html_block" || cur == "html_block") return "\n\n";
  return "  \n";
}

std::string merge_visual_blocks_to_markdown(const json& para_block, const std::string& bucket,
                                            const Ctx& c) {
  std::vector<Seg> segs;
  json blocks = para_block.value("blocks", json::array());
  for (const json* block : blocks_in_index_order(blocks)) {
    auto s = render_visual_block_segments(*block, para_block, bucket, c);
    segs.insert(segs.end(), s.begin(), s.end());
  }
  std::string para_text;
  std::string prev_kind;
  bool first = true;
  for (auto& seg : segs) {
    if (!para_text.empty()) para_text += visual_block_separator(prev_kind, seg.kind);
    para_text += seg.text;
    prev_kind = seg.kind;
    (void)first;
  }
  return para_text;
}

// ---- markdown assembly (lines 361-421) --------------------------------------
std::string mk_blocks_to_markdown(const json& para_blocks, const std::string& make_mode,
                                  const std::string& bucket, const Ctx& c,
                                  std::vector<std::string>& out) {
  for (const auto& para_block : para_blocks) {
    std::string para_text;
    std::string para_type = para_block["type"];
    if (para_type == bt::kText || para_type == bt::kInterlineEquation || para_type == bt::kPhonetic ||
        para_type == bt::kRefText) {
      para_text = merge_para_with_text_img(para_block, c, bucket, c.formula_enable);
    } else if (para_type == bt::kList) {
      for (const auto& block : para_block["blocks"]) {
        std::string item =
            merge_para_with_text_img(block, c, bucket, c.formula_enable, /*escape_prefix=*/false);
        para_text += item + "  \n";
      }
    } else if (para_type == bt::kTitle) {
      int level = get_title_level(para_block);  // title uses defaults (formula_enable=True, img='')
      para_text = std::string(level, '#') + " " + merge_para_with_text(para_block, c);
    } else if (para_type == bt::kImage || para_type == bt::kTable || para_type == bt::kChart) {
      if (make_mode == make_mode::kNlpMd) continue;
      else if (make_mode == make_mode::kMmMd) para_text = merge_visual_blocks_to_markdown(para_block, bucket, c);
    } else if (para_type == bt::kCode) {
      para_text = merge_visual_blocks_to_markdown(para_block, bucket, c);
    }
    if (is_blank(para_text)) continue;
    out.push_back(strip(para_text));
  }
  return {};
}

// ---- get_body_data (lines 765-803) ------------------------------------------
std::pair<std::string, std::string> get_data_from_spans(const json& lines) {
  for (const auto& line : lines)
    for (const auto& span : line.value("spans", json::array())) {
      std::string t = span.value("type", "");
      if (t == ct::kTable) return {get_str(span, "image_path"), get_str(span, "html")};
      if (t == ct::kChart) return {get_str(span, "image_path"), get_str(span, "content")};
      if (t == ct::kImage) return {get_str(span, "image_path"), get_str(span, "content")};
      if (t == ct::kInterlineEquation) return {get_str(span, "image_path"), get_str(span, "content")};
      if (t == ct::kText) return {"", get_str(span, "content")};
    }
  return {"", ""};
}

std::pair<std::string, std::string> get_body_data(const json& para_block) {
  if (para_block.contains("blocks")) {
    for (const auto& block : para_block["blocks"]) {
      std::string bt_ = block.value("type", "");
      if (bt_ == bt::kImageBody || bt_ == bt::kTableBody || bt_ == bt::kChartBody || bt_ == bt::kCodeBody) {
        auto r = get_data_from_spans(block.value("lines", json::array()));
        if (!(r.first.empty() && r.second.empty())) return r;
      }
    }
    return {"", ""};
  }
  return get_data_from_spans(para_block.value("lines", json::array()));
}

// ---- content_list v1 (lines 424-533) ----------------------------------------
void apply_bbox_and_pageidx(json& para_content, const json& para_block, const json& page_size,
                            bool with_page_idx, int page_idx) {
  double pw = page_size[0].get<double>();
  double ph = page_size[1].get<double>();
  if (para_block.contains("bbox") && !para_block["bbox"].is_null()) {
    const json& b = para_block["bbox"];
    double x0 = b[0].get<double>(), y0 = b[1].get<double>(), x1 = b[2].get<double>(), y1 = b[3].get<double>();
    para_content["bbox"] = {(int)(x0 * 1000 / pw), (int)(y0 * 1000 / ph), (int)(x1 * 1000 / pw),
                            (int)(y1 * 1000 / ph)};
  }
  if (with_page_idx) para_content["page_idx"] = page_idx;
}

json make_blocks_to_content_list(const json& para_block, const std::string& bucket, int page_idx,
                                 const json& page_size, const Ctx& c) {
  std::string para_type = para_block["type"];
  json pc = json::object();
  static const std::vector<std::string> text_types = {bt::kText,    bt::kRefText,     bt::kPhonetic,
                                                       bt::kHeader,  bt::kFooter,      bt::kPageNumber,
                                                       bt::kAsideText, bt::kPageFootnote};
  if (std::find(text_types.begin(), text_types.end(), para_type) != text_types.end()) {
    pc = {{"type", para_type}, {"text", merge_para_with_text(para_block, c)}};
  } else if (para_type == bt::kList) {
    pc = {{"type", para_type}, {"sub_type", get_str(para_block, "sub_type")}, {"list_items", json::array()}};
    for (const auto& block : para_block["blocks"]) {
      std::string item = merge_para_with_text(block, c, /*escape_prefix=*/false);
      if (!is_blank(item)) pc["list_items"].push_back(item);
    }
  } else if (para_type == bt::kTitle) {
    int level = get_title_level(para_block);
    pc = {{"type", ct::kText}, {"text", merge_para_with_text(para_block, c)}};
    if (level != 0) pc["text_level"] = level;
  } else if (para_type == bt::kInterlineEquation) {
    pc = {{"type", ct::kEquation}, {"text", merge_para_with_text(para_block, c)}, {"text_format", "latex"}};
  } else if (para_type == bt::kImage) {
    pc = {{"type", ct::kImage}, {"img_path", ""}, {bt::kImageCaption, json::array()}, {bt::kImageFootnote, json::array()}};
    auto [image_path, image_content] = get_body_data(para_block);
    pc["img_path"] = build_media_path(bucket, image_path);
    pc["content"] = image_content;
    if (has_nonempty_str(para_block, "sub_type")) pc["sub_type"] = get_str(para_block, "sub_type");
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kImageCaption) pc[bt::kImageCaption].push_back(merge_para_with_text(block, c));
      if (block["type"] == bt::kImageFootnote) pc[bt::kImageFootnote].push_back(merge_para_with_text(block, c));
    }
  } else if (para_type == bt::kTable) {
    pc = {{"type", ct::kTable}, {"img_path", ""}, {bt::kTableCaption, json::array()}, {bt::kTableFootnote, json::array()}};
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kTableBody) {
        for (const auto& line : block["lines"])
          for (const auto& span : line["spans"])
            if (span["type"] == ct::kTable) {
              if (has_nonempty_str(span, "html"))
                pc[bt::kTableBody] = format_embedded_html(get_str(span, "html"), bucket, c);
              if (has_nonempty_str(span, "image_path"))
                pc["img_path"] = bucket + "/" + get_str(span, "image_path");
            }
      }
      if (block["type"] == bt::kTableCaption) pc[bt::kTableCaption].push_back(merge_para_with_text(block, c));
      if (block["type"] == bt::kTableFootnote) pc[bt::kTableFootnote].push_back(merge_para_with_text(block, c));
    }
  } else if (para_type == bt::kChart) {
    auto [image_path, chart_content] = get_body_data(para_block);
    pc = {{"type", ct::kChart}, {"img_path", build_media_path(bucket, image_path)}, {"content", chart_content},
          {bt::kChartCaption, json::array()}, {bt::kChartFootnote, json::array()}};
    if (has_nonempty_str(para_block, "sub_type")) pc["sub_type"] = get_str(para_block, "sub_type");
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kChartCaption) pc[bt::kChartCaption].push_back(merge_para_with_text(block, c));
      if (block["type"] == bt::kChartFootnote) pc[bt::kChartFootnote].push_back(merge_para_with_text(block, c));
    }
  } else if (para_type == bt::kCode) {
    pc = {{"type", bt::kCode}, {"sub_type", para_block["sub_type"]}, {bt::kCodeCaption, json::array()}};
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kCodeBody) pc[bt::kCodeBody] = render_code_block_markdown(block, para_block, c);
      if (block["type"] == bt::kCodeCaption) pc[bt::kCodeCaption].push_back(merge_para_with_text(block, c));
    }
  }
  apply_bbox_and_pageidx(pc, para_block, page_size, /*with_page_idx=*/true, page_idx);
  return pc;
}

// ---- merge_para_with_text_v2 (lines 806-889) --------------------------------
json merge_para_with_text_v2(const json& para_block) {
  std::string block_text;
  for (const auto& line : para_block["lines"])
    for (const auto& span : line["spans"])
      if (span["type"] == ct::kText) block_text += normalize_text_content(span);
  std::string block_lang = text::detect_lang(block_text);

  json para_content = json::array();
  std::string para_type = para_block["type"];
  const json& lines = para_block["lines"];
  for (size_t i = 0; i < lines.size(); ++i) {
    const json& spans = lines[i]["spans"];
    for (size_t j = 0; j < spans.size(); ++j) {
      const json& span = spans[j];
      // span['content'] is normalized for TEXT (Python mutates it before this loop).
      std::string raw_type = span["type"];
      std::string cur_content =
          (raw_type == ct::kText) ? normalize_text_content(span) : get_str(span, "content");
      if (strip(cur_content).empty()) continue;

      std::string span_type = raw_type;
      if (span_type == ct::kText)
        span_type = (para_type == bt::kPhonetic) ? ctv2::kSpanPhonetic : ctv2::kSpanText;
      if (raw_type == ct::kInlineEquation) span_type = ctv2::kSpanEquationInline;

      if (span_type == ctv2::kSpanText) {
        bool is_last_span = (j == spans.size() - 1);
        bool following = has_following_joinable_span(para_block, i, j);
        std::string span_content;
        if (text::is_cjk_lang(block_lang)) {
          span_content = (following && !is_last_span) ? cur_content + " " : cur_content;
        } else {
          if (is_last_span && text::is_hyphen_at_line_end(cur_content)) {
            bool next_lower = false;
            if (i + 1 < lines.size()) {
              const json& nspans = lines[i + 1].value("spans", json::array());
              if (!nspans.empty() && nspans[0].value("type", "") == ct::kText) {
                std::string nc = get_str(nspans[0], "content");
                if (!nc.empty() && nc[0] >= 'a' && nc[0] <= 'z') next_lower = true;
              }
            }
            span_content = next_lower ? remove_last_codepoint(cur_content) : cur_content;
          } else {
            span_content = following ? cur_content + " " : cur_content;
          }
        }
        if (!para_content.empty() && para_content.back()["type"] == span_type) {
          std::string prev = para_content.back()["content"];
          para_content.back()["content"] = prev + span_content;
        } else {
          para_content.push_back({{"type", span_type}, {"content", span_content}});
        }
      } else if (span_type == ctv2::kSpanPhonetic || span_type == ctv2::kSpanEquationInline) {
        para_content.push_back({{"type", span_type}, {"content", get_str(span, "content")}});
      }
    }
  }
  return para_content;
}

// ---- content_list v2 (lines 536-759) ----------------------------------------
json make_blocks_to_content_list_v2(const json& para_block, const std::string& bucket,
                                    const json& page_size, const Ctx& c) {
  std::string para_type = para_block["type"];
  json pc = json::object();
  if (para_type == bt::kHeader || para_type == bt::kFooter || para_type == bt::kAsideText ||
      para_type == bt::kPageNumber || para_type == bt::kPageFootnote) {
    std::string ctype = para_type == bt::kHeader      ? ctv2::kPageHeader
                        : para_type == bt::kFooter     ? ctv2::kPageFooter
                        : para_type == bt::kAsideText  ? ctv2::kPageAsideText
                        : para_type == bt::kPageNumber ? ctv2::kPageNumber
                                                       : ctv2::kPageFootnote;
    pc = {{"type", ctype}, {"content", {{std::string(ctype) + "_content", merge_para_with_text_v2(para_block)}}}};
  } else if (para_type == bt::kTitle) {
    int level = get_title_level(para_block);
    if (level != 0)
      pc = {{"type", ctv2::kTitle}, {"content", {{"title_content", merge_para_with_text_v2(para_block)}, {"level", level}}}};
    else
      pc = {{"type", ctv2::kParagraph}, {"content", {{"paragraph_content", merge_para_with_text_v2(para_block)}}}};
  } else if (para_type == bt::kText || para_type == bt::kPhonetic) {
    pc = {{"type", ctv2::kParagraph}, {"content", {{"paragraph_content", merge_para_with_text_v2(para_block)}}}};
  } else if (para_type == bt::kInterlineEquation) {
    auto [image_path, math_content] = get_body_data(para_block);
    pc = {{"type", ctv2::kEquationInterline},
          {"content", {{"math_content", math_content}, {"math_type", "latex"}, {"image_source", {{"path", bucket + "/" + image_path}}}}}};
  } else if (para_type == bt::kImage) {
    auto [image_path, image_content] = get_body_data(para_block);
    json caption = json::array(), footnote = json::array();
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kImageCaption) for (auto& s : merge_para_with_text_v2(block)) caption.push_back(s);
      if (block["type"] == bt::kImageFootnote) for (auto& s : merge_para_with_text_v2(block)) footnote.push_back(s);
    }
    pc = {{"type", ctv2::kImage},
          {"content", {{"image_source", {{"path", build_media_path(bucket, image_path)}}}, {"content", image_content},
                       {"image_caption", caption}, {"image_footnote", footnote}}}};
    if (has_nonempty_str(para_block, "sub_type")) pc["sub_type"] = get_str(para_block, "sub_type");
  } else if (para_type == bt::kTable) {
    auto [image_path, html] = get_body_data(para_block);
    std::string table_html = format_embedded_html(html, bucket, c);
    json caption = json::array(), footnote = json::array();
    int count_table = 0;
    for (size_t p = table_html.find("<table"); p != std::string::npos; p = table_html.find("<table", p + 1)) ++count_table;
    int nest = count_table > 1 ? 2 : 1;
    bool complex_ = table_html.find("colspan") != std::string::npos ||
                    table_html.find("rowspan") != std::string::npos || nest > 1;
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kTableCaption) for (auto& s : merge_para_with_text_v2(block)) caption.push_back(s);
      if (block["type"] == bt::kTableFootnote) for (auto& s : merge_para_with_text_v2(block)) footnote.push_back(s);
    }
    pc = {{"type", ctv2::kTable},
          {"content", {{"image_source", {{"path", bucket + "/" + image_path}}}, {"table_caption", caption},
                       {"table_footnote", footnote}, {"html", table_html},
                       {"table_type", complex_ ? ctv2::kTableComplex : ctv2::kTableSimple}, {"table_nest_level", nest}}}};
  } else if (para_type == bt::kChart) {
    auto [image_path, chart_content] = get_body_data(para_block);
    json caption = json::array(), footnote = json::array();
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kChartCaption) for (auto& s : merge_para_with_text_v2(block)) caption.push_back(s);
      if (block["type"] == bt::kChartFootnote) for (auto& s : merge_para_with_text_v2(block)) footnote.push_back(s);
    }
    pc = {{"type", ctv2::kChart},
          {"content", {{"image_source", {{"path", build_media_path(bucket, image_path)}}}, {"content", chart_content},
                       {"chart_caption", caption}, {"chart_footnote", footnote}}}};
    if (has_nonempty_str(para_block, "sub_type")) pc["sub_type"] = get_str(para_block, "sub_type");
  } else if (para_type == bt::kCode) {
    json caption = json::array(), code_content = json::array();
    for (const auto& block : para_block["blocks"]) {
      if (block["type"] == bt::kCodeCaption) for (auto& s : merge_para_with_text_v2(block)) caption.push_back(s);
      if (block["type"] == bt::kCodeBody) code_content = merge_para_with_text_v2(block);
    }
    std::string sub_type = para_block["sub_type"];
    if (sub_type == bt::kCode)
      pc = {{"type", ctv2::kCode}, {"content", {{"code_caption", caption}, {"code_content", code_content}, {"code_language", get_str(para_block, "guess_lang", "txt")}}}};
    else if (sub_type == bt::kAlgorithm)
      pc = {{"type", ctv2::kAlgorithm}, {"content", {{"algorithm_caption", caption}, {"algorithm_content", code_content}}}};
    else
      throw std::runtime_error("Unknown code sub_type: " + sub_type);
  } else if (para_type == bt::kRefText) {
    pc = {{"type", ctv2::kList},
          {"content", {{"list_type", ctv2::kListRef}, {"list_items", json::array({{{"item_type", "text"}, {"item_content", merge_para_with_text_v2(para_block)}}})}}}};
  } else if (para_type == bt::kList) {
    std::string list_type = ctv2::kListText;
    if (para_block.contains("sub_type")) {
      std::string st = para_block["sub_type"];
      if (st == bt::kRefText) list_type = ctv2::kListRef;
      else if (st == bt::kText) list_type = ctv2::kListText;
      else throw std::runtime_error("Unknown list sub_type: " + st);
    }
    json list_items = json::array();
    for (const auto& block : para_block["blocks"]) {
      json ic = merge_para_with_text_v2(block);
      if (!ic.empty()) list_items.push_back({{"item_type", "text"}, {"item_content", ic}});
    }
    pc = {{"type", ctv2::kList}, {"content", {{"list_type", list_type}, {"list_items", list_items}}}};
  }
  apply_bbox_and_pageidx(pc, para_block, page_size, /*with_page_idx=*/false, 0);
  return pc;
}

}  // namespace

json union_make(const json& pdf_info, const std::string& make_mode, const std::string& bucket,
                bool formula_enable, bool table_enable, const LatexDelimiters& delim) {
  Ctx c{delim, formula_enable, table_enable};
  const std::string& MM = make_mode::kMmMd;
  const std::string& NLP = make_mode::kNlpMd;
  const std::string& CL = make_mode::kContentList;
  const std::string& CLV2 = make_mode::kContentListV2;

  if (make_mode == MM || make_mode == NLP) {
    std::vector<std::string> out;
    for (const auto& page : pdf_info) {
      const json& layout = page.value("para_blocks", json::array());
      if (layout.empty()) continue;
      mk_blocks_to_markdown(layout, make_mode, bucket, c, out);
    }
    std::string joined;
    for (size_t i = 0; i < out.size(); ++i) { if (i) joined += "\n\n"; joined += out[i]; }
    return joined;  // JSON string
  }
  if (make_mode == CL) {
    json out = json::array();
    for (const auto& page : pdf_info) {
      json para_blocks = json::array();
      for (const auto& b : page.value("para_blocks", json::array())) para_blocks.push_back(b);
      for (const auto& b : page.value("discarded_blocks", json::array())) para_blocks.push_back(b);
      if (para_blocks.empty()) continue;
      int page_idx = page.value("page_idx", 0);
      const json& page_size = page["page_size"];
      for (const auto& pb : para_blocks)
        out.push_back(make_blocks_to_content_list(pb, bucket, page_idx, page_size, c));
    }
    return out;
  }
  if (make_mode == CLV2) {
    json out = json::array();
    for (const auto& page : pdf_info) {
      json para_blocks = json::array();
      for (const auto& b : page.value("para_blocks", json::array())) para_blocks.push_back(b);
      for (const auto& b : page.value("discarded_blocks", json::array())) para_blocks.push_back(b);
      const json& page_size = page["page_size"];
      json page_contents = json::array();
      for (const auto& pb : para_blocks)
        page_contents.push_back(make_blocks_to_content_list_v2(pb, bucket, page_size, c));
      out.push_back(page_contents);
    }
    return out;
  }
  return nullptr;
}

}  // namespace mineru

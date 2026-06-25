// Copyright (c) mlx-mineru.
// Cross-page table merge — faithful port of MinerU mineru/utils/table_merge.py.
// Scope: the deterministic pipeline path (equal-column / repeated-header / continuation
// merges, caption & footnote gating, colspan reconciliation). The `cell_merge` semantic
// merge (VLM-only field) and the cross-page rowspan blank-clip helpers are intentionally
// not ported — pipeline tables never carry `cell_merge`, and clip/carry only fire when a
// rowspan straddles the page boundary; both are documented in REVIEW.md.
#include "mineru/table_merge.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace mineru {
namespace {

using json = nlohmann::json;

constexpr int MAX_HEADER_ROWS = 5;

// ---- text utilities (full_to_half, entity unescape, continuation marker) ------------------

// Convert full-width chars (U+FF01..U+FF5E) to ASCII by subtracting 0xFEE0, like full_to_half.
std::string full_to_half(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0, n = s.size();
  while (i < n) {
    unsigned char c = (unsigned char)s[i];
    uint32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 0x6 && i + 1 < n) { cp = c & 0x1F; len = 2; }
    else if ((c >> 4) == 0xE && i + 2 < n) { cp = c & 0x0F; len = 3; }
    else if ((c >> 3) == 0x1E && i + 3 < n) { cp = c & 0x07; len = 4; }
    else { out += (char)c; ++i; continue; }
    for (int k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    if (cp >= 0xFF01 && cp <= 0xFF5E) {
      out += (char)(cp - 0xFEE0);  // shifts into printable ASCII (single byte)
    } else {
      out.append(s, i, len);
    }
    i += len;
  }
  return out;
}

// Minimal HTML entity unescape for cell text comparison (bs4 get_text() unescapes).
std::string unescape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '&') {
      size_t sc = s.find(';', i);
      if (sc != std::string::npos && sc - i <= 8) {
        std::string e = s.substr(i + 1, sc - i - 1);
        std::string rep;
        if (e == "amp") rep = "&";
        else if (e == "lt") rep = "<";
        else if (e == "gt") rep = ">";
        else if (e == "quot") rep = "\"";
        else if (e == "apos" || e == "#39") rep = "'";
        else if (e == "nbsp") rep = "\xC2\xA0";
        if (!rep.empty()) { out += rep; i = sc + 1; continue; }
      }
    }
    out += s[i++];
  }
  return out;
}

std::string lower(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}
std::string strip(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}
// Remove all whitespace (≈ "".join(text.split())).
std::string remove_ws(const std::string& s) {
  std::string o;
  for (char c : s) if (c != ' ' && c != '\t' && c != '\r' && c != '\n') o += c;
  return o;
}

// Faithful is_table_continuation_text: markers copied verbatim from MinerU.
bool is_table_continuation_text(const std::string& text) {
  std::string t = lower(full_to_half(strip(text)));
  if (t.empty()) return false;
  static const std::vector<std::string> END = {
      "(\xE7\xBB\xAD)", "(\xE7\xBB\xAD\xE8\xA1\xA8)", "(\xE7\xBB\xAD\xE4\xB8\x8A\xE8\xA1\xA8)",
      "(continued)", "(cont.)", "(cont\xE2\x80\x99""d)", "(\xE2\x80\xA6""continued)",
      "continued", "\xE7\xBB\xAD\xE8\xA1\xA8"};
  for (const std::string& m : END) {
    std::string ml = lower(m);
    if (t.size() >= ml.size() && t.compare(t.size() - ml.size(), ml.size(), ml) == 0) {
      if (ml == "continued") {
        size_t start = t.size() - ml.size();
        if (start == 0 || !std::isalpha((unsigned char)t[start - 1])) return true;
      } else {
        return true;
      }
    }
  }
  return t.find("(continued)") != std::string::npos;
}

// ---- lightweight mutable HTML table model -------------------------------------------------

struct Cell {
  std::string tag;  // "td" / "th"
  std::vector<std::pair<std::string, std::string>> attrs;
  std::string inner;  // raw inner HTML

  int attr_int(const std::string& name, int def) const {
    for (auto& a : attrs) if (a.first == name) { return a.second.empty() ? def : std::atoi(a.second.c_str()); }
    return def;
  }
  int colspan() const { return attr_int("colspan", 1); }
  int rowspan() const { return attr_int("rowspan", 1); }
  void set_attr(const std::string& name, const std::string& val) {
    for (auto& a : attrs) if (a.first == name) { a.second = val; return; }
    attrs.emplace_back(name, val);
  }
  // bs4 get_text(): strip nested tags + unescape entities.
  std::string text() const {
    std::string t;
    bool intag = false;
    for (char c : inner) {
      if (c == '<') intag = true;
      else if (c == '>') intag = false;
      else if (!intag) t += c;
    }
    return unescape(t);
  }
  std::string serialize() const {
    std::string s = "<" + tag;
    for (auto& a : attrs) s += " " + a.first + "=\"" + a.second + "\"";
    return s + ">" + inner + "</" + tag + ">";
  }
};

struct Row {
  std::vector<std::pair<std::string, std::string>> attrs;
  std::vector<Cell> cells;
  std::string serialize() const {
    std::string s = "<tr";
    for (auto& a : attrs) s += " " + a.first + "=\"" + a.second + "\"";
    s += ">";
    for (auto& c : cells) s += c.serialize();
    return s + "</tr>";
  }
};

struct Table {
  bool has_tbody = false;
  std::vector<std::pair<std::string, std::string>> table_attrs;
  std::vector<Row> rows;
  std::string serialize() const {
    std::string s = "<table";
    for (auto& a : table_attrs) s += " " + a.first + "=\"" + a.second + "\"";
    s += ">";
    if (has_tbody) s += "<tbody>";
    for (auto& r : rows) s += r.serialize();
    if (has_tbody) s += "</tbody>";
    return s + "</table>";
  }
};

std::vector<std::pair<std::string, std::string>> parse_attrs(const std::string& s) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r' || s[i] == '/')) ++i;
    if (i >= s.size()) break;
    size_t ns = i;
    while (i < s.size() && s[i] != '=' && s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r') ++i;
    std::string name = s.substr(ns, i - ns);
    std::string val;
    if (i < s.size() && s[i] == '=') {
      ++i;
      if (i < s.size() && (s[i] == '"' || s[i] == '\'')) {
        char q = s[i++];
        size_t vs = i;
        while (i < s.size() && s[i] != q) ++i;
        val = s.substr(vs, i - vs);
        if (i < s.size()) ++i;
      } else {
        size_t vs = i;
        while (i < s.size() && s[i] != ' ' && s[i] != '\t') ++i;
        val = s.substr(vs, i - vs);
      }
    }
    if (!name.empty()) out.emplace_back(name, val);
  }
  return out;
}

// Parse a `<table>..</table>` (with optional <tbody>) into the model. Cells preserved verbatim.
bool parse_table(const std::string& html, Table& t) {
  size_t tp = html.find("<table");
  if (tp == std::string::npos) return false;
  size_t topen = html.find('>', tp);
  if (topen == std::string::npos) return false;
  t.table_attrs = parse_attrs(html.substr(tp + 6, topen - (tp + 6)));
  t.has_tbody = html.find("<tbody", topen) != std::string::npos &&
                html.find("<tbody", topen) < html.find("</table", topen);
  size_t end = html.rfind("</table");
  if (end == std::string::npos) end = html.size();
  size_t p = topen + 1;
  while (true) {
    size_t tr = html.find("<tr", p);
    if (tr == std::string::npos || tr >= end) break;
    size_t tro = html.find('>', tr);
    if (tro == std::string::npos) break;
    Row row;
    row.attrs = parse_attrs(html.substr(tr + 3, tro - (tr + 3)));
    size_t trend = html.find("</tr", tro);
    if (trend == std::string::npos) trend = end;
    size_t cp = tro + 1;
    while (true) {
      size_t lt = std::string::npos;
      bool is_td = false;
      size_t ctd = html.find("<td", cp), cth = html.find("<th", cp);
      if (ctd != std::string::npos && ctd < trend && (cth == std::string::npos || ctd < cth)) { lt = ctd; is_td = true; }
      else if (cth != std::string::npos && cth < trend) { lt = cth; is_td = false; }
      if (lt == std::string::npos || lt >= trend) break;
      size_t co = html.find('>', lt);
      if (co == std::string::npos) break;
      Cell cell;
      cell.tag = is_td ? "td" : "th";
      cell.attrs = parse_attrs(html.substr(lt + 3, co - (lt + 3)));
      std::string close = is_td ? "</td" : "</th";
      size_t ce = html.find(close, co);
      if (ce == std::string::npos) ce = trend;
      cell.inner = html.substr(co + 1, ce - (co + 1));
      row.cells.push_back(std::move(cell));
      size_t after = html.find('>', ce);
      cp = (after == std::string::npos) ? ce + 4 : after + 1;
    }
    t.rows.push_back(std::move(row));
    p = trend + 4;
  }
  return true;
}

// ---- occupancy scan (faithful _scan_rows) -------------------------------------------------

struct RowMetrics {
  int row_idx = 0, effective_cols = 0, actual_cols = 0, visual_cols = 0;
};
struct RowScan {
  std::vector<int> row_effective_cols;
  std::vector<RowMetrics> row_metrics;
  int total_cols = 0;
  int last_nonempty = -1;  // index into row_metrics, or -1
  std::map<int, std::set<int>> tail_occupied;
};

RowScan scan_rows(const std::vector<Row>& rows,
                  const std::map<int, std::set<int>>* initial_occupied = nullptr,
                  int start_row_idx = 0) {
  std::map<int, std::set<int>> occupied;
  int max_cols = 0;
  if (initial_occupied) {
    for (auto& kv : *initial_occupied) {
      if (kv.second.empty()) continue;
      occupied[kv.first] = kv.second;
      max_cols = std::max(max_cols, *kv.second.rbegin() + 1);
    }
  }
  RowScan r;
  for (int local = 0; local < (int)rows.size(); ++local) {
    std::set<int>& occ_row = occupied[local];
    int col_idx = 0, actual = 0;
    const std::vector<Cell>& cells = rows[local].cells;
    for (const Cell& cell : cells) {
      while (occ_row.count(col_idx)) ++col_idx;
      int cs = cell.colspan(), rs = cell.rowspan();
      actual += cs;
      for (int ro = 0; ro < rs; ++ro)
        for (int c = col_idx; c < col_idx + cs; ++c) occupied[local + ro].insert(c);
      col_idx += cs;
      max_cols = std::max(max_cols, col_idx);
    }
    int eff = occ_row.empty() ? 0 : (*occ_row.rbegin() + 1);
    r.row_effective_cols.push_back(eff);
    max_cols = std::max(max_cols, eff);
    RowMetrics m{start_row_idx + local, eff, actual, (int)cells.size()};
    r.row_metrics.push_back(m);
    if (!cells.empty()) r.last_nonempty = (int)r.row_metrics.size() - 1;
  }
  r.total_cols = max_cols;
  int n = (int)rows.size();
  for (auto& kv : occupied)
    if (kv.first >= n && !kv.second.empty()) r.tail_occupied[kv.first - n] = kv.second;
  return r;
}

// occupied[col] = (source_row, source_cell) for one target row (_scan_row_visual_sources).
std::pair<std::map<int, std::pair<int, int>>, int> scan_row_visual_sources(
    const std::vector<Row>& rows, int target, const std::map<int, std::set<int>>* initial = nullptr) {
  if (target < 0) target += (int)rows.size();
  if (target < 0 || target >= (int)rows.size()) return {{}, 0};
  std::map<int, std::map<int, std::pair<int, int>>> occupied;
  int total = 0;
  if (initial) {
    for (auto& kv : *initial) {
      if (kv.second.empty()) continue;
      for (int c : kv.second) occupied[kv.first][c] = {-1, c};
      total = std::max(total, *kv.second.rbegin() + 1);
    }
  }
  for (int r = 0; r <= target; ++r) {
    std::map<int, std::pair<int, int>>& occ_row = occupied[r];
    int col_idx = 0;
    const std::vector<Cell>& cells = rows[r].cells;
    for (int ci = 0; ci < (int)cells.size(); ++ci) {
      while (occ_row.count(col_idx)) ++col_idx;
      int cs = cells[ci].colspan(), rs = cells[ci].rowspan();
      for (int ro = 0; ro < rs; ++ro)
        for (int c = col_idx; c < col_idx + cs; ++c) occupied[r + ro][c] = {r, ci};
      col_idx += cs;
      total = std::max(total, col_idx);
    }
  }
  return {occupied[target], total};
}

int rendered_segments(const std::vector<Row>& rows, int target) {
  auto [occ, total] = scan_row_visual_sources(rows, target);
  if (total == 0) return 0;
  int count = 0;
  std::pair<int, int> prev{-2, -2};
  bool have_prev = false;
  for (int c = 0; c < total; ++c) {
    auto it = occ.find(c);
    if (it == occ.end()) { have_prev = false; continue; }
    if (!have_prev || it->second != prev) { ++count; prev = it->second; have_prev = true; }
  }
  return count;
}

// ---- row signatures + header detection ----------------------------------------------------

struct RowSignature {
  int effective_cols = 0;
  std::vector<int> colspans, rowspans;
  std::vector<std::string> normalized, display;
  int cell_count() const { return (int)colspans.size(); }
};

RowSignature build_signature(const Row& row, int eff) {
  RowSignature s;
  s.effective_cols = eff;
  for (const Cell& c : row.cells) {
    s.colspans.push_back(c.colspan());
    s.rowspans.push_back(c.rowspan());
    s.normalized.push_back(remove_ws(full_to_half(c.text())));
    s.display.push_back(full_to_half(strip(c.text())));
  }
  return s;
}

struct TableState {
  json* owner_block = nullptr;  // the table block
  json* body_span = nullptr;    // span holding "html"
  Table table;
  int total_cols = 0;
  std::vector<RowSignature> front_header_info;
  std::vector<RowMetrics> front_first_data_row_metrics;
  RowMetrics last_data;  // valid if last_data_valid
  bool last_data_valid = false;
  std::vector<int> row_effective_cols;
  std::map<int, std::set<int>> tail_occupied;
  bool dirty = false;
};

void build_front_cache(const std::vector<Row>& rows, std::vector<RowSignature>& header_info,
                       std::vector<RowMetrics>& first_data) {
  int front_limit = std::min((int)rows.size(), MAX_HEADER_ROWS + 1);
  std::vector<Row> front(rows.begin(), rows.begin() + front_limit);
  RowScan scan = scan_rows(front);
  header_info.clear();
  for (int i = 0; i < std::min((int)front.size(), MAX_HEADER_ROWS); ++i)
    header_info.push_back(build_signature(front[i], scan.row_effective_cols[i]));
  first_data = scan.row_metrics;
}

void refresh_state_metrics(TableState& s) {
  RowScan scan = scan_rows(s.table.rows);
  s.row_effective_cols = scan.row_effective_cols;
  s.total_cols = scan.total_cols;
  s.last_data_valid = scan.last_nonempty >= 0;
  if (s.last_data_valid) s.last_data = scan.row_metrics[scan.last_nonempty];
  s.tail_occupied = scan.tail_occupied;
  build_front_cache(s.table.rows, s.front_header_info, s.front_first_data_row_metrics);
}

// structural (then visual) header detection between two states. Returns header_count.
int detect_table_headers(const TableState& s1, const TableState& s2) {
  int min_rows = std::min({(int)s1.front_header_info.size(), (int)s2.front_header_info.size(), MAX_HEADER_ROWS});
  int header_rows = 0;
  for (int i = 0; i < min_rows; ++i) {
    const RowSignature& a = s1.front_header_info[i];
    const RowSignature& b = s2.front_header_info[i];
    bool match = a.cell_count() == b.cell_count() && a.effective_cols == b.effective_cols &&
                 a.colspans == b.colspans && a.rowspans == b.rowspans && a.normalized == b.normalized;
    if (match) ++header_rows;
    else break;
  }
  if (header_rows > 0) return header_rows;
  // visual fallback: text + rendered-segment equality
  header_rows = 0;
  for (int i = 0; i < min_rows; ++i) {
    const RowSignature& a = s1.front_header_info[i];
    const RowSignature& b = s2.front_header_info[i];
    if (a.normalized == b.normalized &&
        rendered_segments(s1.table.rows, i) == rendered_segments(s2.table.rows, i))
      ++header_rows;
    else break;
  }
  return header_rows;
}

int expand_header_count_by_rowspan(const std::vector<Row>& rows, int header_count) {
  if (header_count <= 0 || rows.empty()) return header_count;
  int expanded = std::min(header_count, (int)rows.size());
  for (int ri = 0; ri < expanded; ++ri)
    for (const Cell& c : rows[ri].cells) {
      int rs = c.rowspan();
      if (rs > 1) expanded = std::min((int)rows.size(), std::max(expanded, ri + rs));
    }
  return expanded;
}

bool check_rows_match(const TableState& prev, const TableState& cur) {
  if (!prev.last_data_valid) return false;
  int header_count = detect_table_headers(prev, cur);
  header_count = expand_header_count_by_rowspan(cur.table.rows, header_count);
  if (header_count < 0 || header_count >= (int)cur.front_first_data_row_metrics.size()) return false;
  const RowMetrics& first_data = cur.front_first_data_row_metrics[header_count];
  int prev_seg = rendered_segments(prev.table.rows, prev.last_data.row_idx);
  int cur_seg = rendered_segments(cur.table.rows, first_data.row_idx);
  return prev.last_data.effective_cols == first_data.effective_cols ||
         prev.last_data.actual_cols == first_data.actual_cols || prev_seg == cur_seg;
}

// ---- block helpers ------------------------------------------------------------------------

json* find_body_block(json& table_block) {
  for (json& b : table_block["blocks"])
    if (b.value("type", "") == "table_body") return &b;
  return nullptr;
}
json* find_body_span(json& table_block) {
  json* body = find_body_block(table_block);
  if (body && body->contains("lines") && !(*body)["lines"].empty() &&
      !(*body)["lines"][0]["spans"].empty())
    return &(*body)["lines"][0]["spans"][0];
  return nullptr;
}
std::string caption_text(const json& cap) {
  std::string s;
  for (const json& line : cap.value("lines", json::array()))
    for (const json& span : line.value("spans", json::array())) {
      std::string c = span.value("content", "");
      if (!c.empty()) s += (s.empty() ? "" : " ") + c;
    }
  return s;
}
bool is_continuation_caption(const json& cap) { return is_table_continuation_text(caption_text(cap)); }

bool is_post_table_non_continuation_caption(json& table_block, const json& cap) {
  if (is_continuation_caption(cap)) return false;
  json* body = find_body_block(table_block);
  if (!body || !body->contains("bbox") || !cap.contains("bbox")) return false;
  return cap["bbox"][1].get<double>() >= (*body)["bbox"][3].get<double>();
}

// ---- can_merge / perform_merge ------------------------------------------------------------

bool can_merge_tables(TableState& cur, TableState& prev) {
  json& cur_block = *cur.owner_block;
  json& prev_block = *prev.owner_block;
  int footnote_count = 0;
  for (const json& b : prev_block["blocks"]) if (b.value("type", "") == "table_footnote") ++footnote_count;
  std::vector<const json*> merge_caps;
  for (json& b : cur_block["blocks"])
    if (b.value("type", "") == "table_caption" && !is_post_table_non_continuation_caption(cur_block, b))
      merge_caps.push_back(&b);
  if (!merge_caps.empty()) {
    bool cont = false;
    for (const json* c : merge_caps) if (is_continuation_caption(*c)) cont = true;
    if (!cont) return false;
    if (footnote_count > 1) return false;
  } else if (footnote_count > 0) {
    return false;
  }
  double x0a = cur_block["bbox"][0], x1a = cur_block["bbox"][2];
  double x0b = prev_block["bbox"][0], x1b = prev_block["bbox"][2];
  double w1 = x1a - x0a, w2 = x1b - x0b;
  if (std::min(w1, w2) <= 0 || std::abs(w1 - w2) / std::min(w1, w2) >= 0.1) return false;
  if (prev.total_cols == cur.total_cols) return true;
  return check_rows_match(prev, cur);
}

// adjust_table_rows_colspan — reconcile a narrower table's rows to target_cols.
int calc_row_columns(const Row& row) {
  int n = 0;
  for (const Cell& c : row.cells) n += c.colspan();
  return n;
}
void adjust_rows_colspan(std::vector<Row>& rows, int start, int end,
                         const std::vector<int>& row_eff, const std::vector<int>& ref_structure,
                         int ref_visual_cols, int target_cols, const Row& ref_row) {
  for (int ri = start; ri < end; ++ri) {
    Row& row = rows[ri];
    if (row.cells.empty()) continue;
    int cur_eff = row_eff[ri], cur_cols = calc_row_columns(row);
    if (cur_eff >= target_cols || cur_cols >= target_cols) continue;
    bool same_visual = (int)row.cells.size() == ref_visual_cols;
    bool cols_match = row.cells.size() == ref_row.cells.size();
    if (cols_match)
      for (size_t k = 0; k < row.cells.size(); ++k)
        if (row.cells[k].colspan() != ref_row.cells[k].colspan()) { cols_match = false; break; }
    if (same_visual && cols_match) {
      if (row.cells.size() <= ref_structure.size())
        for (size_t ci = 0; ci < row.cells.size(); ++ci)
          if (ci < ref_structure.size() && ref_structure[ci] > 1)
            row.cells[ci].set_attr("colspan", std::to_string(ref_structure[ci]));
    } else {
      int diff = target_cols - cur_eff;
      if (diff > 0) {
        Cell& last = row.cells.back();
        last.set_attr("colspan", std::to_string(last.colspan() + diff));
      }
    }
  }
}

void perform_table_merge(TableState& prev, TableState& cur, std::vector<json>& wait_footnotes) {
  int header_count = detect_table_headers(prev, cur);
  header_count = expand_header_count_by_rowspan(cur.table.rows, header_count);
  std::vector<Row>& rows1 = prev.table.rows;
  std::vector<Row>& rows2 = cur.table.rows;
  bool previous_adjusted = false;
  if (!rows1.empty() && !rows2.empty() && header_count < (int)rows2.size()) {
    Row& last1 = rows1.back();
    Row& first2 = rows2[header_count];
    int cols1 = prev.total_cols, cols2 = cur.total_cols;
    if (cols1 > cols2) {
      std::vector<int> ref;
      for (const Cell& c : last1.cells) ref.push_back(c.colspan());
      adjust_rows_colspan(rows2, header_count, (int)rows2.size(), cur.row_effective_cols, ref,
                          (int)last1.cells.size(), cols1, first2);
    } else if (cols2 > cols1) {
      std::vector<int> ref;
      for (const Cell& c : first2.cells) ref.push_back(c.colspan());
      adjust_rows_colspan(rows1, 0, (int)rows1.size(), prev.row_effective_cols, ref,
                          (int)first2.cells.size(), cols2, last1);
      previous_adjusted = true;
    }
  }
  if (previous_adjusted) refresh_state_metrics(prev);
  // append data rows (header skipped) verbatim
  for (int i = header_count; i < (int)rows2.size(); ++i) rows1.push_back(rows2[i]);
  refresh_state_metrics(prev);
  // footnotes: drop previous footnotes, re-add carried ones with cross_page + reindexed
  json& prev_block = *prev.owner_block;
  json kept = json::array();
  for (json& b : prev_block["blocks"]) if (b.value("type", "") != "table_footnote") kept.push_back(b);
  json* body = find_body_block(prev_block);
  int body_index = (body && body->contains("index") && (*body)["index"].is_number())
                       ? (*body)["index"].get<int>() : 0;
  bool have_body_index = body && body->contains("index") && (*body)["index"].is_number();
  int offset = 1;
  for (json fn : wait_footnotes) {
    fn["cross_page"] = true;
    if (have_body_index) fn["index"] = body_index + offset;
    else fn.erase("index");
    kept.push_back(fn);
    ++offset;
  }
  prev_block["blocks"] = kept;
  prev.dirty = true;
}

// Restore captions mis-attached below the table as standalone text blocks on the current page.
void restore_post_table_captions(json& page_info, json& table_block,
                                 const std::vector<json>& caption_blocks) {
  if (caption_blocks.empty()) return;
  json& para = page_info["para_blocks"];
  int insert_idx = -1;
  for (int i = 0; i < (int)para.size(); ++i)
    if (&para[i] == &table_block) { insert_idx = i + 1; break; }
  if (insert_idx < 0) return;
  // build set of caption contents to remove (compare by serialized content)
  std::vector<json> restored;
  for (const json& cap : caption_blocks) {
    json t = cap;
    t["type"] = "text";
    restored.push_back(t);
  }
  for (int k = (int)restored.size() - 1; k >= 0; --k)
    para.insert(para.begin() + insert_idx, restored[k]);
  // remove the captions from the table block
  json kept = json::array();
  for (json& b : table_block["blocks"]) {
    bool drop = false;
    for (const json& cap : caption_blocks)
      if (b == cap) { drop = true; break; }
    if (!drop) kept.push_back(b);
  }
  table_block["blocks"] = kept;
}

bool build_state(json& table_block, TableState& s) {
  json* span = find_body_span(table_block);
  if (!span || !span->contains("html")) return false;
  std::string html = span->value("html", "");
  if (html.empty()) return false;
  if (!parse_table(html, s.table)) return false;
  s.owner_block = &table_block;
  s.body_span = span;
  RowScan scan = scan_rows(s.table.rows);
  s.total_cols = scan.total_cols;
  s.row_effective_cols = scan.row_effective_cols;
  s.tail_occupied = scan.tail_occupied;
  s.last_data_valid = scan.last_nonempty >= 0;
  if (s.last_data_valid) s.last_data = scan.row_metrics[scan.last_nonempty];
  build_front_cache(s.table.rows, s.front_header_info, s.front_first_data_row_metrics);
  return true;
}

}  // namespace

void cross_page_table_merge(json& pdf_info) {
  const char* env = std::getenv("MINERU_TABLE_MERGE_ENABLE");
  if (env) {
    std::string v = lower(env);
    if (v == "false" || v == "0" || v == "no") return;
  }
  if (!pdf_info.is_array()) return;

  // State per table block, keyed by address; merged-away current blocks excluded from final serialize.
  std::map<json*, TableState> states;
  std::set<json*> merged_away;

  auto get_state = [&](json& block) -> TableState* {
    auto it = states.find(&block);
    if (it != states.end()) return &it->second;
    TableState s;
    if (!build_state(block, s)) return nullptr;
    auto [ins, ok] = states.emplace(&block, std::move(s));
    return &ins->second;
  };

  for (int page_idx = (int)pdf_info.size() - 1; page_idx >= 1; --page_idx) {
    json& page = pdf_info[page_idx];
    json& prev_page = pdf_info[page_idx - 1];
    if (!page.contains("para_blocks") || page["para_blocks"].empty()) continue;
    if (!prev_page.contains("para_blocks") || prev_page["para_blocks"].empty()) continue;
    json& cur_block = page["para_blocks"][0];
    json& prev_block = prev_page["para_blocks"].back();
    if (cur_block.value("type", "") != "table" || prev_block.value("type", "") != "table") continue;

    TableState* cur = get_state(cur_block);
    TableState* prev = get_state(prev_block);
    if (!cur || !prev) continue;

    std::vector<json> post_caps;
    for (json& b : cur_block["blocks"])
      if (b.value("type", "") == "table_caption" && is_post_table_non_continuation_caption(cur_block, b))
        post_caps.push_back(b);
    std::vector<json> wait_footnotes;
    for (json& b : cur_block["blocks"])
      if (b.value("type", "") == "table_footnote") wait_footnotes.push_back(b);

    if (!can_merge_tables(*cur, *prev)) continue;

    perform_table_merge(*prev, *cur, wait_footnotes);
    // Serialize the grown table now, locating its body span fresh (perform_table_merge
    // reassigns prev_block["blocks"], so any pre-merge span pointer is stale). For chains the
    // in-memory state keeps accumulating rows, so re-serializing each round is idempotent.
    if (json* span = find_body_span(prev_block)) (*span)["html"] = prev->table.serialize();
    restore_post_table_captions(page, cur_block, post_caps);

    merged_away.insert(&cur_block);
    for (json& b : cur_block["blocks"]) {
      b["lines"] = json::array();
      b["lines_deleted"] = true;
    }
  }
  (void)merged_away;
}

}  // namespace mineru

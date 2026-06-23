// Copyright (c) mlx-mineru.
#include "mineru/otsl.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace mineru {
namespace {

const std::string NL = "<nl>", FCEL = "<fcel>", ECEL = "<ecel>", LCEL = "<lcel>",
                  UCEL = "<ucel>", XCEL = "<xcel>";

bool is_token(const std::string& s) {
  return s == NL || s == FCEL || s == ECEL || s == LCEL || s == UCEL || s == XCEL;
}

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n\f\v");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n\f\v");
  return s.substr(a, b - a + 1);
}

std::string html_escape(const std::string& s) {  // html.escape(quote=True)
  std::string out;
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&#x27;"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

struct Cell {
  int row_span = 1, col_span = 1;
  int sr = 0, er = 1, sc = 0, ec = 1;
  std::string text;
  bool real = false;  // an actual (non-filler) cell
};

// Split content into the ordered token list and the mixed (token|text) list.
void extract_tokens_and_text(const std::string& s, std::vector<std::string>& tokens,
                             std::vector<std::string>& mixed) {
  static const std::vector<std::string> toks = {NL, FCEL, ECEL, LCEL, UCEL, XCEL};
  size_t i = 0;
  size_t last = 0;
  while (i < s.size()) {
    bool matched = false;
    for (const auto& t : toks) {
      if (s.compare(i, t.size(), t) == 0) {
        if (i > last) {  // preceding text
          std::string seg = s.substr(last, i - last);
          if (!strip(seg).empty()) mixed.push_back(seg);
        }
        tokens.push_back(t);
        mixed.push_back(t);
        i += t.size();
        last = i;
        matched = true;
        break;
      }
    }
    if (!matched) ++i;
  }
  if (last < s.size()) {
    std::string seg = s.substr(last);
    if (!strip(seg).empty()) mixed.push_back(seg);
  }
}

int count_right(const std::vector<std::vector<std::string>>& rows, int c, int r,
                bool lcel_xcel) {
  int span = 0, ci = c;
  while (ci < (int)rows[r].size()) {
    const std::string& t = rows[r][ci];
    bool in = lcel_xcel ? (t == LCEL || t == XCEL) : (t == UCEL || t == XCEL);
    if (!in) break;
    ++ci;
    ++span;
  }
  return span;
}

int count_down(const std::vector<std::vector<std::string>>& rows, int c, int r, bool ucel_xcel) {
  int span = 0, ri = r;
  while (ri < (int)rows.size() && c < (int)rows[ri].size()) {
    const std::string& t = rows[ri][c];
    bool in = ucel_xcel ? (t == UCEL || t == XCEL) : (t == LCEL || t == XCEL);
    if (!in) break;
    ++ri;
    ++span;
  }
  return span;
}

}  // namespace

std::string convert_otsl_to_html(const std::string& otsl_content) {
  if (otsl_content.size() >= 15 && otsl_content.compare(0, 6, "<table") == 0 &&
      otsl_content.compare(otsl_content.size() - 8, 8, "</table>") == 0)
    return otsl_content;
  if (strip(otsl_content).empty()) return "";

  std::vector<std::string> tokens, mixed;
  extract_tokens_and_text(otsl_content, tokens, mixed);

  // Group tokens into rows (split by <nl>).
  std::vector<std::vector<std::string>> rows;
  std::vector<std::string> cur;
  for (const auto& t : tokens) {
    if (t == NL) { rows.push_back(cur); cur.clear(); }
    else cur.push_back(t);
  }
  if (!cur.empty()) rows.push_back(cur);
  if (rows.empty()) return "";

  // Pad rows to max_cols with <ecel>.
  size_t max_cols = 0;
  for (auto& r : rows) max_cols = std::max(max_cols, r.size());
  for (auto& r : rows) while (r.size() < max_cols) r.push_back(ECEL);

  // Rebuild `texts` with padding (mirrors otsl_parse_texts).
  std::vector<std::string> texts;
  size_t ti = 0;
  for (auto& row : rows) {
    for (auto& token : row) {
      texts.push_back(token);
      if (ti < mixed.size() && mixed[ti] == token) {
        ++ti;
        if (ti < mixed.size() && !is_token(mixed[ti])) { texts.push_back(mixed[ti]); ++ti; }
      }
    }
    texts.push_back(NL);
    if (ti < mixed.size() && mixed[ti] == NL) ++ti;
  }

  // Build cells.
  std::vector<Cell> cells;
  int r_idx = 0, c_idx = 0;
  for (size_t i = 0; i < texts.size(); ++i) {
    const std::string& text = texts[i];
    if (text == FCEL || text == ECEL) {
      int row_span = 1, col_span = 1, right_offset = 1;
      std::string cell_text;
      if (text != ECEL && i + 1 < texts.size() && !is_token(texts[i + 1])) {
        cell_text = texts[i + 1];
        right_offset = 2;
      }
      std::string next_right = (i + right_offset < texts.size()) ? texts[i + right_offset] : "";
      std::string next_bottom;
      if (r_idx + 1 < (int)rows.size() && c_idx < (int)rows[r_idx + 1].size())
        next_bottom = rows[r_idx + 1][c_idx];
      if (next_right == LCEL || next_right == XCEL)
        col_span += count_right(rows, c_idx + 1, r_idx, /*lcel_xcel=*/true);
      if (next_bottom == UCEL || next_bottom == XCEL)
        row_span += count_down(rows, c_idx, r_idx + 1, /*ucel_xcel=*/true);
      Cell c;
      c.text = strip(cell_text);
      c.row_span = row_span;
      c.col_span = col_span;
      c.sr = r_idx;
      c.er = r_idx + row_span;
      c.sc = c_idx;
      c.ec = c_idx + col_span;
      c.real = true;
      cells.push_back(c);
    }
    if (is_token(text) && text != NL) ++c_idx;
    if (text == NL) { ++r_idx; c_idx = 0; }
  }

  int nrows = (int)rows.size(), ncols = (int)max_cols;
  if (cells.empty()) return "";

  // Grid of cell pointers (-1 = filler).
  std::vector<std::vector<int>> grid(nrows, std::vector<int>(ncols, -1));
  for (int ci = 0; ci < (int)cells.size(); ++ci) {
    const Cell& c = cells[ci];
    for (int i = std::min(c.sr, nrows); i < std::min(c.er, nrows); ++i)
      for (int j = std::min(c.sc, ncols); j < std::min(c.ec, ncols); ++j)
        grid[i][j] = ci;
  }

  std::string out = "<table>";
  for (int i = 0; i < nrows; ++i) {
    out += "<tr>";
    for (int j = 0; j < ncols; ++j) {
      int ci = grid[i][j];
      // Empty filler cell (never overwritten) -> default empty td at its own origin.
      Cell def;
      const Cell& c = (ci >= 0) ? cells[ci] : def;
      int sr = (ci >= 0) ? c.sr : i, sc = (ci >= 0) ? c.sc : j;
      if (sr != i || sc != j) continue;  // spanned-over cell
      std::string content = html_escape(strip(c.text));
      std::string open = "<td";
      if (c.row_span > 1) open += " rowspan=\"" + std::to_string(c.row_span) + "\"";
      if (c.col_span > 1) open += " colspan=\"" + std::to_string(c.col_span) + "\"";
      open += ">";
      out += open + content + "</td>";
    }
    out += "</tr>";
  }
  out += "</table>";
  return out;
}

}  // namespace mineru

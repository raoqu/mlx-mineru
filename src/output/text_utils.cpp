// Copyright (c) mlx-mineru.
#include "text_utils.hpp"

namespace mineru::text {

// --- UTF-8 -------------------------------------------------------------------
std::vector<char32_t> utf8_decode(const std::string& s) {
  std::vector<char32_t> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
    else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
    else { cp = c; len = 1; }  // invalid lead byte: pass through
    if (i + len > n) { out.push_back(c); ++i; continue; }
    for (int k = 1; k < len; ++k) {
      unsigned char cc = static_cast<unsigned char>(s[i + k]);
      cp = (cp << 6) | (cc & 0x3F);
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void utf8_append(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string utf8_encode(const std::vector<char32_t>& cps) {
  std::string out;
  for (char32_t cp : cps) utf8_append(out, cp);
  return out;
}

// --- char_utils.py -----------------------------------------------------------
std::string full_to_half_exclude_marks(const std::string& text) {
  std::string out;
  for (char32_t code : utf8_decode(text)) {
    if ((0xFF21 <= code && code <= 0xFF3A) || (0xFF41 <= code && code <= 0xFF5A) ||
        (0xFF10 <= code && code <= 0xFF19)) {
      utf8_append(out, code - 0xFEE0);
    } else {
      utf8_append(out, code);
    }
  }
  return out;
}

static bool is_ascii_space(char32_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool is_hyphen_at_line_end(const std::string& line) {
  // Regex equivalent: [A-Za-z]+[-­‐‑⁃]\s*$
  static const char32_t kHyphens[] = {0x2D, 0xAD, 0x2010, 0x2011, 0x2043};
  std::vector<char32_t> cps = utf8_decode(line);
  // Drop trailing whitespace.
  size_t end = cps.size();
  while (end > 0 && is_ascii_space(cps[end - 1])) --end;
  if (end < 2) return false;
  char32_t last = cps[end - 1];
  bool is_hyphen = false;
  for (char32_t h : kHyphens)
    if (last == h) { is_hyphen = true; break; }
  if (!is_hyphen) return false;
  char32_t prev = cps[end - 2];
  return (prev >= 'A' && prev <= 'Z') || (prev >= 'a' && prev <= 'z');
}

// --- language.py (approximation) ---------------------------------------------
std::string detect_lang(const std::string& text) {
  if (text.empty()) return "";
  bool has_kana = false, has_hangul = false, has_han = false;
  for (char32_t c : utf8_decode(text)) {
    if ((0x3040 <= c && c <= 0x309F) || (0x30A0 <= c && c <= 0x30FF)) has_kana = true;
    else if ((0xAC00 <= c && c <= 0xD7A3) || (0x1100 <= c && c <= 0x11FF)) has_hangul = true;
    else if ((0x4E00 <= c && c <= 0x9FFF) || (0x3400 <= c && c <= 0x4DBF) ||
             (0x20000 <= c && c <= 0x2A6DF) || (0xF900 <= c && c <= 0xFAFF))
      has_han = true;
  }
  if (has_kana) return "ja";
  if (has_hangul) return "ko";
  if (has_han) return "zh";
  return "en";
}

bool is_cjk_lang(const std::string& lang) {
  return lang == "zh" || lang == "ja" || lang == "ko";
}

// --- markdown_utils.py -------------------------------------------------------
std::string escape_conservative_markdown_text(const std::string& content) {
  if (content.empty()) return content;
  static const std::string kSpecials = "*_`~$";
  std::string out;
  int preceding_backslashes = 0;
  // ASCII-only specials & backslash; iterating bytes matches code-point iteration
  // because UTF-8 continuation bytes never equal these and reset the counter.
  for (char ch : content) {
    if (ch == '\\') {
      out.push_back(ch);
      ++preceding_backslashes;
      continue;
    }
    if (kSpecials.find(ch) != std::string::npos && preceding_backslashes % 2 == 0) {
      out.push_back('\\');
    }
    out.push_back(ch);
    preceding_backslashes = 0;
  }
  return out;
}

std::string escape_text_block_markdown_prefix(const std::string& content) {
  if (content.empty()) return content;
  // ^(?P<indent>[ \t]{0,3})(?P<marker>#{1,6}|[+-])(?=[ \t])
  size_t i = 0, n = content.size();
  size_t indent = 0;
  while (i < n && (content[i] == ' ' || content[i] == '\t') && indent < 3) { ++i; ++indent; }
  if (i >= n) return content;
  size_t marker_start = i;
  if (content[i] == '#') {
    size_t hashes = 0;
    while (i < n && content[i] == '#' && hashes < 6) { ++i; ++hashes; }
  } else if (content[i] == '+' || content[i] == '-') {
    ++i;
  } else {
    return content;
  }
  // Lookahead (?=[ \t])
  if (i >= n || (content[i] != ' ' && content[i] != '\t')) return content;
  return content.substr(0, marker_start) + "\\" + content.substr(marker_start);
}

std::string html_escape_no_quote(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char ch : s) {
    switch (ch) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      default: out.push_back(ch);
    }
  }
  return out;
}

static std::string normalize_text_content(const nlohmann::json& span) {
  if (!span.contains("content") || span["content"].is_null()) return "";
  return full_to_half_exclude_marks(span["content"].get<std::string>());
}

static bool ends_with_ws(const std::string& s) {
  if (s.empty()) return false;
  char c = s.back();
  return c == ' ' || c == '\n' || c == '\t';
}

std::string render_algorithm_html_from_lines(const nlohmann::json& lines,
                                             const std::string& inline_left,
                                             const std::string& inline_right) {
  std::vector<std::string> parts;
  std::string previous_span_type;
  if (lines.is_array()) {
    for (const auto& line : lines) {
      for (const auto& span : line.value("spans", nlohmann::json::array())) {
        std::string span_type = span.value("type", "");
        if (span_type == "text") {
          std::string content = normalize_text_content(span);
          parts.push_back(html_escape_no_quote(content));
          if (!content.empty()) previous_span_type = span_type;
        } else if (span_type == "inline_equation") {
          std::string content = span.contains("content") && !span["content"].is_null()
                                    ? span["content"].get<std::string>()
                                    : "";
          // strip() check
          std::string stripped = content;
          // simple ASCII/space strip for emptiness test
          size_t a = stripped.find_first_not_of(" \t\n\r\f\v");
          if (a != std::string::npos) {
            if (previous_span_type == "inline_equation" && !parts.empty() &&
                !ends_with_ws(parts.back())) {
              parts.push_back(" ");
            }
            parts.push_back(inline_left + html_escape_no_quote(content) + inline_right);
            previous_span_type = span_type;
          }
        }
      }
    }
  }
  std::string body;
  for (const auto& p : parts) body += p;
  // strip() for emptiness
  if (body.find_first_not_of(" \t\n\r\f\v") == std::string::npos) return "";
  return "<div class=\"mineru-algorithm\" style=\"white-space: pre-wrap; font-family:monospace;\">\n" +
         body + "\n</div>";
}

}  // namespace mineru::text

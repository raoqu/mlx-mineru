// Copyright (c) mlx-mineru.
#include "mineru/post_process.hpp"

#include <functional>
#include <regex>
#include <vector>

namespace mineru::pp {
namespace {

// Replace all non-overlapping matches of `re` using a transform on each match.
std::string replace_fn(const std::string& s, const std::regex& re,
                       const std::function<std::string(const std::smatch&)>& fn) {
  std::string out;
  auto begin = std::sregex_iterator(s.begin(), s.end(), re);
  auto end = std::sregex_iterator();
  size_t last = 0;
  for (auto it = begin; it != end; ++it) {
    const std::smatch& m = *it;
    out += s.substr(last, m.position() - last);
    out += fn(m);
    last = m.position() + m.length();
  }
  out += s.substr(last);
  return out;
}

std::string strip(const std::string& s) {
  const char* ws = " \t\r\n\f\v";
  size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

// UTF-8 decode (for the en-dash / numeric check).
std::vector<char32_t> decode(const std::string& s) {
  std::vector<char32_t> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    unsigned char c = (unsigned char)s[i];
    char32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
    else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
    else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
    else { cp = c; len = 1; }
    for (int k = 1; k < len && i + k < n; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    out.push_back(cp);
    i += len;
  }
  return out;
}

// fullmatch of [–\d\-,\s]+  (en-dash U+2013, digits, '-', ',', whitespace), >=1 char.
bool is_numeric_dash(const std::string& s) {
  auto cps = decode(s);
  if (cps.empty()) return false;
  for (char32_t c : cps) {
    bool ok = (c == 0x2013) || (c >= '0' && c <= '9') || c == '-' || c == ',' || c == ' ' ||
              c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    if (!ok) return false;
  }
  return true;
}

}  // namespace

std::string try_convert_display_to_inline(const std::string& text) {
  static const std::regex re(R"(\\\[([\s\S]*?)\\\])");
  return replace_fn(text, re, [](const std::smatch& m) {
    std::string inner = m[1].str();
    return is_numeric_dash(inner) ? "\\[" + inner + "\\]" : "\\(" + inner + "\\)";
  });
}

std::string try_fix_macro_spacing_in_markdown(const std::string& text) {
  static const std::vector<std::string> known = {"\\top", "\\int", "\\inf"};
  static const std::vector<std::string> targets = {"\\cong", "\\to", "\\times", "\\subset", "\\in"};
  auto fix_inner = [](std::string inner) {
    for (const auto& macro : targets) {
      std::string esc;
      for (char c : macro) { if (!std::isalnum((unsigned char)c)) esc += '\\'; esc += c; }
      std::regex re(esc + R"(([a-zA-Z])(?![a-zA-Z]))");
      inner = replace_fn(inner, re, [&](const std::smatch& m) {
        std::string cand = macro + m[1].str();
        for (const auto& k : known) if (k == cand) return m[0].str();
        return macro + " " + m[1].str();
      });
    }
    return inner;
  };
  // split by (\(...\)) keeping the formula chunks; process inner of each.
  static const std::regex fre(R"(\\\([\s\S]*?\\\))");
  return replace_fn(text, fre, [&](const std::smatch& m) {
    std::string part = m[0].str();
    std::string inner = part.substr(2, part.size() - 4);
    return "\\(" + fix_inner(inner) + "\\)";
  });
}

std::string try_move_underscores_outside(const std::string& text) {
  static const std::regex re(R"(\\\(([\s\S]+?)\\\))");
  static const std::regex usplit(R"(_{3,})");
  return replace_fn(text, re, [](const std::smatch& m) -> std::string {
    std::string inner = m[1].str();
    // split keeping the underscore separators.
    std::vector<std::string> parts;
    auto begin = std::sregex_iterator(inner.begin(), inner.end(), usplit);
    auto end = std::sregex_iterator();
    size_t last = 0;
    for (auto it = begin; it != end; ++it) {
      parts.push_back(inner.substr(last, it->position() - last));
      parts.push_back(it->str());
      last = it->position() + it->length();
    }
    parts.push_back(inner.substr(last));
    if (parts.size() == 1) return m[0].str();  // no 3+ underscores
    std::vector<std::string> result;
    for (const auto& p : parts) {
      if (!p.empty() && std::regex_match(p, usplit))
        result.push_back(p);
      else if (!strip(p).empty())
        result.push_back("\\(" + p + "\\)");
    }
    std::string joined;
    for (size_t i = 0; i < result.size(); ++i) { if (i) joined += " "; joined += result[i]; }
    return joined;
  });
}

std::string process_text(const std::string& text) {
  return try_move_underscores_outside(try_fix_macro_spacing_in_markdown(
      try_convert_display_to_inline(text)));
}

std::string try_fix_equation_delimiters(const std::string& latex) {
  std::string s = strip(latex);
  if (s.size() >= 2 && s.compare(0, 2, "\\[") == 0) s = s.substr(2);
  if (s.size() >= 2 && s.compare(s.size() - 2, 2, "\\]") == 0) s = s.substr(0, s.size() - 2);
  return strip(s);
}

std::string try_fix_unbalanced_braces(const std::string& f) {
  std::vector<size_t> stack;
  std::vector<bool> unmatched(f.size(), false);
  for (size_t i = 0; i < f.size(); ++i) {
    if (f[i] == '{' || f[i] == '}') {
      int bs = 0;
      long j = (long)i - 1;
      while (j >= 0 && f[j] == '\\') { ++bs; --j; }
      if (bs % 2 == 1) continue;  // escaped brace
      if (f[i] == '{') stack.push_back(i);
      else { if (!stack.empty()) stack.pop_back(); else unmatched[i] = true; }
    }
  }
  for (size_t idx : stack) unmatched[idx] = true;
  std::string out;
  for (size_t i = 0; i < f.size(); ++i) if (!unmatched[i]) out.push_back(f[i]);
  return out;
}

std::string process_equation(const std::string& latex) {
  return try_fix_unbalanced_braces(try_fix_equation_delimiters(latex));
}

}  // namespace mineru::pp

// Copyright (c) mlx-mineru.
// Faithful C++ ports of MinerU's text helpers used by the output renderer:
//   - mineru/utils/char_utils.py  (full_to_half_exclude_marks, is_hyphen_at_line_end)
//   - mineru/backend/utils/markdown_utils.py (escape_*; render_algorithm_html_from_lines)
//   - mineru/utils/language.py    (detect_lang) — approximated by CJK-script detection
//     (see note below). Python operates on Unicode code points; we decode UTF-8.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace mineru::text {

// --- UTF-8 <-> code points ---------------------------------------------------
std::vector<char32_t> utf8_decode(const std::string& s);
std::string utf8_encode(const std::vector<char32_t>& cps);
void utf8_append(std::string& out, char32_t cp);

// --- char_utils.py -----------------------------------------------------------
// Convert full-width ASCII letters/digits (FF21-FF3A, FF41-FF5A, FF10-FF19) to
// half-width; leave everything else (incl. full-width marks) untouched.
std::string full_to_half_exclude_marks(const std::string& text);

// True if the line ends (ignoring trailing whitespace) with an ASCII word
// followed by a line-break hyphen char (- ­ ‐ ‑ ⁃).
bool is_hyphen_at_line_end(const std::string& line);

// --- language.py (approximation) ---------------------------------------------
// MinerU uses fast-langdetect (fasttext lid.176). The renderer only branches on
// membership in {zh, ja, ko}, so we classify by script: Hiragana/Katakana -> ja,
// Hangul -> ko, otherwise Han -> zh, otherwise "en". This agrees with
// fast-langdetect on linguistically unambiguous text; revisit if a real port is
// needed (see PLAN.md risks).
std::string detect_lang(const std::string& text);
bool is_cjk_lang(const std::string& lang);  // lang in {zh, ja, ko}

// --- markdown_utils.py -------------------------------------------------------
std::string escape_conservative_markdown_text(const std::string& content);
std::string escape_text_block_markdown_prefix(const std::string& content);
std::string html_escape_no_quote(const std::string& s);  // & < > only

// Render an algorithm block's inline spans to indentation-preserving HTML.
std::string render_algorithm_html_from_lines(const nlohmann::json& lines,
                                             const std::string& inline_left,
                                             const std::string& inline_right);

}  // namespace mineru::text

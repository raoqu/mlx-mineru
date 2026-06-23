// Copyright (c) mlx-mineru.
#include "mineru/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "unicode_tables.hpp"

namespace mineru {
namespace {
using nlohmann::json;

// --- UTF-8 helpers ----------------------------------------------------------
std::vector<char32_t> utf8_decode(const std::string& s) {
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
    if (i + len > n) { out.push_back(c); ++i; continue; }
    for (int k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    out.push_back(cp);
    i += len;
  }
  return out;
}
void utf8_append(std::string& out, char32_t cp) {
  if (cp < 0x80) out.push_back((char)cp);
  else if (cp < 0x800) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// GPT-2 bytes<->unicode mapping.
struct ByteUnicode {
  char32_t b2u[256];
  std::unordered_map<char32_t, uint8_t> u2b;
  ByteUnicode() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    std::vector<char> in_bs(256, 0);
    for (int b : bs) in_bs[b] = 1;
    for (int b = 0; b < 256; ++b) {
      if (!in_bs[b]) { bs.push_back(b); cs.push_back(256 + n); ++n; }
    }
    for (size_t k = 0; k < bs.size(); ++k) {
      b2u[bs[k]] = (char32_t)cs[k];
      u2b[(char32_t)cs[k]] = (uint8_t)bs[k];
    }
  }
};
const ByteUnicode& byte_unicode() {
  static ByteUnicode bu;
  return bu;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("tokenizer: cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool is_crlf(char32_t c) { return c == '\r' || c == '\n'; }

}  // namespace

struct Qwen2Tokenizer::Impl {
  std::unordered_map<std::string, int> vocab;      // token string (byte-level) -> id
  std::vector<std::string> id_to_token;            // id -> token string
  std::unordered_map<std::string, int> merge_rank; // "A B" -> rank
  // Special/added tokens.
  std::vector<std::pair<std::string, int>> specials;  // content -> id, longest-first
  std::unordered_map<int, std::string> special_id_to_content;
  std::unordered_map<std::string, int> special_content_to_id;

  // Ordered-first-match Qwen2 pre-tokenizer; returns end index of token at i.
  size_t match_pretoken(const std::vector<char32_t>& c, size_t i) const;
  std::vector<int> bpe_word(const std::string& piece_utf8) const;
};

using ucd::is_letter;
using ucd::is_number;
using ucd::is_whitespace;

size_t Qwen2Tokenizer::Impl::match_pretoken(const std::vector<char32_t>& c, size_t i) const {
  size_t n = c.size();
  auto L = [&](size_t k) { return k < n && is_letter(c[k]); };
  auto N = [&](size_t k) { return k < n && is_number(c[k]); };
  auto W = [&](size_t k) { return k < n && is_whitespace(c[k]); };

  // (a) (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (c[i] == '\'') {
    auto low = [&](char32_t x) { return (x >= 'A' && x <= 'Z') ? x + 32 : x; };
    if (i + 2 < n) {
      char32_t a = low(c[i + 1]), b = low(c[i + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    if (i + 1 < n) {
      char32_t a = low(c[i + 1]);
      if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
    }
  }
  // (b) [^\r\n\p{L}\p{N}]?\p{L}+
  {
    size_t j = i;
    if (!is_crlf(c[j]) && !L(j) && !N(j)) {  // optional leading non-letter/number
      if (L(j + 1)) { ++j; }                 // only take it if a letter follows
    }
    if (L(j)) {
      ++j;
      while (L(j)) ++j;
      return j;
    }
  }
  // (c) \p{N}  (single number codepoint)
  if (N(i)) return i + 1;
  // (d)  ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    size_t j = i;
    if (c[j] == ' ') {
      // optional space, but only if a valid symbol run follows
      size_t k = j + 1;
      if (k < n && !W(k) && !L(k) && !N(k)) ++j;
    }
    if (j < n && !W(j) && !L(j) && !N(j)) {
      ++j;
      while (j < n && !W(j) && !L(j) && !N(j)) ++j;
      while (j < n && is_crlf(c[j])) ++j;
      return j;
    }
  }
  // (e) \s*[\r\n]+
  if (W(i)) {
    size_t r = i;
    while (r < n && W(r)) ++r;                  // maximal whitespace run [i,r)
    size_t last_crlf = std::string::npos;
    for (size_t k = i; k < r; ++k)
      if (is_crlf(c[k])) last_crlf = k;
    if (last_crlf != std::string::npos) return last_crlf + 1;  // ends at last CRLF
    // (f) \s+(?!\S)
    if (r == n) return r;          // run at end of string
    if (r - 1 > i) return r - 1;   // all but the last whitespace (kept for next word)
    // (g) \s+
    return r;
  }
  // Fallback: consume one codepoint (should not normally happen).
  return i + 1;
}

std::vector<int> Qwen2Tokenizer::Impl::bpe_word(const std::string& piece_utf8) const {
  // Byte-level: each byte -> a mapped unicode char (as its own symbol).
  const ByteUnicode& bu = byte_unicode();
  std::vector<std::string> syms;
  for (unsigned char ch : piece_utf8) {
    std::string s;
    utf8_append(s, bu.b2u[ch]);
    syms.push_back(std::move(s));
  }
  if (syms.empty()) return {};

  // GPT-2 BPE: repeatedly merge the single lowest-rank adjacent pair.
  while (syms.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = std::string::npos;
    for (size_t k = 0; k + 1 < syms.size(); ++k) {
      auto it = merge_rank.find(syms[k] + " " + syms[k + 1]);
      if (it != merge_rank.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = k;
      }
    }
    if (best_i == std::string::npos) break;
    // Merge all non-overlapping occurrences of that exact pair.
    const std::string& a = syms[best_i];
    std::string b = syms[best_i + 1];
    std::vector<std::string> merged;
    for (size_t k = 0; k < syms.size();) {
      if (k + 1 < syms.size() && syms[k] == a && syms[k + 1] == b) {
        merged.push_back(a + b);
        k += 2;
      } else {
        merged.push_back(syms[k]);
        k += 1;
      }
    }
    syms.swap(merged);
  }

  std::vector<int> ids;
  for (auto& s : syms) {
    auto it = vocab.find(s);
    if (it == vocab.end())
      throw std::runtime_error("tokenizer: BPE symbol not in vocab: " + s);
    ids.push_back(it->second);
  }
  return ids;
}

Qwen2Tokenizer::Qwen2Tokenizer() : impl_(std::make_unique<Impl>()) {}
Qwen2Tokenizer::~Qwen2Tokenizer() = default;
Qwen2Tokenizer::Qwen2Tokenizer(Qwen2Tokenizer&&) noexcept = default;
Qwen2Tokenizer& Qwen2Tokenizer::operator=(Qwen2Tokenizer&&) noexcept = default;

Qwen2Tokenizer Qwen2Tokenizer::load(const std::string& dir) {
  Qwen2Tokenizer tok;
  Impl& m = *tok.impl_;

  // vocab.json
  json vocab = json::parse(read_file(dir + "/vocab.json"));
  int max_id = 0;
  for (auto& [k, v] : vocab.items()) {
    int id = v.get<int>();
    m.vocab[k] = id;
    max_id = std::max(max_id, id);
  }

  // added/special tokens from tokenizer.json
  json tj = json::parse(read_file(dir + "/tokenizer.json"));
  if (tj.contains("added_tokens")) {
    for (auto& t : tj["added_tokens"]) {
      int id = t["id"].get<int>();
      std::string content = t["content"].get<std::string>();
      m.special_id_to_content[id] = content;
      m.special_content_to_id[content] = id;
      m.specials.push_back({content, id});
      max_id = std::max(max_id, id);
    }
  }
  // Longest content first so we match greedily.
  std::sort(m.specials.begin(), m.specials.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  m.id_to_token.assign(max_id + 1, "");
  for (auto& [k, id] : m.vocab) m.id_to_token[id] = k;
  for (auto& [id, content] : m.special_id_to_content) m.id_to_token[id] = content;

  // merges.txt (skip an optional "#version" header line)
  {
    std::istringstream in(read_file(dir + "/merges.txt"));
    std::string line;
    int rank = 0;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line[0] == '#') continue;
      m.merge_rank[line] = rank++;  // line is already "A B"
    }
  }
  return tok;
}

std::vector<int> Qwen2Tokenizer::encode(const std::string& text) const {
  const Impl& m = *impl_;
  std::vector<int> out;

  // Split off special tokens (literal, longest-first), BPE the rest.
  size_t pos = 0;
  while (pos < text.size()) {
    size_t next = std::string::npos;
    size_t match_len = 0;
    int match_id = -1;
    // find the earliest special-token occurrence at or after pos
    for (auto& [content, id] : m.specials) {
      size_t f = text.find(content, pos);
      if (f != std::string::npos && (next == std::string::npos || f < next ||
                                     (f == next && content.size() > match_len))) {
        next = f;
        match_len = content.size();
        match_id = id;
      }
    }
    size_t normal_end = (next == std::string::npos) ? text.size() : next;
    if (normal_end > pos) {
      std::string segment = text.substr(pos, normal_end - pos);
      std::vector<char32_t> cps = utf8_decode(segment);
      size_t i = 0;
      while (i < cps.size()) {
        size_t e = m.match_pretoken(cps, i);
        if (e <= i) e = i + 1;
        std::string piece;
        for (size_t k = i; k < e; ++k) utf8_append(piece, cps[k]);
        for (int id : m.bpe_word(piece)) out.push_back(id);
        i = e;
      }
    }
    if (next == std::string::npos) break;
    out.push_back(match_id);
    pos = next + match_len;
  }
  return out;
}

std::string Qwen2Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  const Impl& m = *impl_;
  const ByteUnicode& bu = byte_unicode();
  std::string out;
  std::string byte_buf;  // accumulated byte-level chars from normal tokens

  auto flush = [&]() {
    if (byte_buf.empty()) return;
    // Map byte-level unicode chars back to raw bytes, then it's UTF-8.
    std::string raw;
    for (char32_t cp : utf8_decode(byte_buf)) {
      auto it = bu.u2b.find(cp);
      if (it != bu.u2b.end()) raw.push_back((char)it->second);
    }
    out += raw;
    byte_buf.clear();
  };

  for (int id : ids) {
    auto sp = m.special_id_to_content.find(id);
    if (sp != m.special_id_to_content.end()) {
      flush();
      if (!skip_special) out += sp->second;
      continue;
    }
    if (id >= 0 && id < (int)m.id_to_token.size()) byte_buf += m.id_to_token[id];
  }
  flush();
  return out;
}

int Qwen2Tokenizer::token_to_id(const std::string& token) const {
  auto it = impl_->vocab.find(token);
  return it == impl_->vocab.end() ? -1 : it->second;
}
int Qwen2Tokenizer::special_id(const std::string& content) const {
  auto it = impl_->special_content_to_id.find(content);
  return it == impl_->special_content_to_id.end() ? -1 : it->second;
}
int Qwen2Tokenizer::vocab_size() const { return (int)impl_->id_to_token.size(); }

}  // namespace mineru

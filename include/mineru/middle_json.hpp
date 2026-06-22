// Copyright (c) mlx-mineru.
// Typed model of MinerU's `middle_json` — the central data contract that every
// backend produces and every output renderer consumes.
//
// Design: fields we know about are typed for ergonomic access in later phases;
// every other key is preserved verbatim in `extra` so the round-trip is lossless
// (field-level compatible) even as MinerU adds fields. Serialization is
// semantically equal to the input (key order may differ; values do not).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace mineru {

using Bbox = std::array<double, 4>;

// A span: the smallest content unit (a run of text, an inline formula, an image,
// a table HTML blob, etc.). See ContentType for `type`.
struct Span {
  std::string type;
  std::optional<Bbox> bbox;
  std::optional<std::string> content;     // text / LaTeX / (table) HTML
  std::optional<std::string> image_path;  // for image / interline_equation spans
  std::optional<std::string> html;        // table HTML (when stored separately)
  nlohmann::json extra;                    // any other keys, preserved verbatim
};

// A line: a horizontal run grouping spans.
struct Line {
  std::optional<Bbox> bbox;
  std::vector<Span> spans;
  nlohmann::json extra;
};

// A block: a paragraph/region. Composite blocks (image/table/chart/code) carry
// child blocks (body + captions + footnotes) in `blocks`.
struct Block {
  std::string type;
  std::optional<Bbox> bbox;
  std::optional<int> index;  // reading order
  std::optional<int> level;  // title heading level
  std::optional<std::vector<Line>> lines;
  std::optional<std::vector<Block>> blocks;
  nlohmann::json extra;
};

// A page of the document.
struct Page {
  std::optional<int> page_idx;
  std::optional<std::vector<double>> page_size;  // [w, h]
  std::vector<Block> para_blocks;
  std::vector<Block> discarded_blocks;
  std::optional<std::vector<Block>> preproc_blocks;
  nlohmann::json extra;
};

// The whole document intermediate representation.
struct MiddleJson {
  std::vector<Page> pdf_info;
  std::optional<std::string> backend;       // "_backend"
  std::optional<std::string> version_name;  // "_version_name"
  nlohmann::json extra;
};

// nlohmann (de)serialization — declared here, defined in core/middle_json.cpp.
void to_json(nlohmann::json& j, const Span& v);
void from_json(const nlohmann::json& j, Span& v);
void to_json(nlohmann::json& j, const Line& v);
void from_json(const nlohmann::json& j, Line& v);
void to_json(nlohmann::json& j, const Block& v);
void from_json(const nlohmann::json& j, Block& v);
void to_json(nlohmann::json& j, const Page& v);
void from_json(const nlohmann::json& j, Page& v);
void to_json(nlohmann::json& j, const MiddleJson& v);
void from_json(const nlohmann::json& j, MiddleJson& v);

// Convenience: parse a middle.json document / dump it with MinerU's formatting
// (`ensure_ascii=False, indent=4` equivalent).
MiddleJson parse_middle_json(const std::string& text);
std::string dump_middle_json(const MiddleJson& doc, int indent = 4);

}  // namespace mineru

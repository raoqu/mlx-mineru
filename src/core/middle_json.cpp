// Copyright (c) mlx-mineru.
#include "mineru/middle_json.hpp"

namespace mineru {
namespace {

using nlohmann::json;

// Pull a known key out of `rest` (so it won't land in `extra`) and, if present,
// decode it into `out`.
template <typename T>
void take(json& rest, const char* key, T& out) {
  auto it = rest.find(key);
  if (it == rest.end()) return;
  out = it->get<T>();
  rest.erase(it);
}

template <typename T>
void take_opt(json& rest, const char* key, std::optional<T>& out) {
  auto it = rest.find(key);
  if (it == rest.end() || it->is_null()) return;
  out = it->get<T>();
  rest.erase(it);
}

// Write an optional only when it holds a value.
template <typename T>
void put_opt(json& j, const char* key, const std::optional<T>& v) {
  if (v.has_value()) j[key] = *v;
}

// Start a serialized object from the catch-all `extra` (which never contains a
// known key, because parsing removed them). Known keys are then layered on top.
json base(const json& extra) {
  return extra.is_object() ? extra : json::object();
}

}  // namespace

void to_json(json& j, const Span& v) {
  j = base(v.extra);
  j["type"] = v.type;
  put_opt(j, "bbox", v.bbox);
  put_opt(j, "content", v.content);
  put_opt(j, "image_path", v.image_path);
  put_opt(j, "html", v.html);
}

void from_json(const json& j, Span& v) {
  json rest = j;
  take(rest, "type", v.type);
  take_opt(rest, "bbox", v.bbox);
  take_opt(rest, "content", v.content);
  take_opt(rest, "image_path", v.image_path);
  take_opt(rest, "html", v.html);
  v.extra = std::move(rest);
}

void to_json(json& j, const Line& v) {
  j = base(v.extra);
  put_opt(j, "bbox", v.bbox);
  j["spans"] = v.spans;
}

void from_json(const json& j, Line& v) {
  json rest = j;
  take_opt(rest, "bbox", v.bbox);
  take(rest, "spans", v.spans);
  v.extra = std::move(rest);
}

void to_json(json& j, const Block& v) {
  j = base(v.extra);
  j["type"] = v.type;
  put_opt(j, "bbox", v.bbox);
  put_opt(j, "index", v.index);
  put_opt(j, "level", v.level);
  put_opt(j, "lines", v.lines);
  put_opt(j, "blocks", v.blocks);
}

void from_json(const json& j, Block& v) {
  json rest = j;
  take(rest, "type", v.type);
  take_opt(rest, "bbox", v.bbox);
  take_opt(rest, "index", v.index);
  take_opt(rest, "level", v.level);
  take_opt(rest, "lines", v.lines);
  take_opt(rest, "blocks", v.blocks);
  v.extra = std::move(rest);
}

void to_json(json& j, const Page& v) {
  j = base(v.extra);
  put_opt(j, "page_idx", v.page_idx);
  put_opt(j, "page_size", v.page_size);
  j["para_blocks"] = v.para_blocks;
  j["discarded_blocks"] = v.discarded_blocks;
  put_opt(j, "preproc_blocks", v.preproc_blocks);
}

void from_json(const json& j, Page& v) {
  json rest = j;
  take_opt(rest, "page_idx", v.page_idx);
  take_opt(rest, "page_size", v.page_size);
  if (rest.contains("para_blocks")) take(rest, "para_blocks", v.para_blocks);
  if (rest.contains("discarded_blocks")) take(rest, "discarded_blocks", v.discarded_blocks);
  take_opt(rest, "preproc_blocks", v.preproc_blocks);
  v.extra = std::move(rest);
}

void to_json(json& j, const MiddleJson& v) {
  j = base(v.extra);
  j["pdf_info"] = v.pdf_info;
  put_opt(j, "_backend", v.backend);
  put_opt(j, "_version_name", v.version_name);
}

void from_json(const json& j, MiddleJson& v) {
  json rest = j;
  take(rest, "pdf_info", v.pdf_info);
  take_opt(rest, "_backend", v.backend);
  take_opt(rest, "_version_name", v.version_name);
  v.extra = std::move(rest);
}

MiddleJson parse_middle_json(const std::string& text) {
  return json::parse(text).get<MiddleJson>();
}

std::string dump_middle_json(const MiddleJson& doc, int indent) {
  // ensure_ascii=False equivalent: nlohmann dump() with ensure_ascii=false.
  return json(doc).dump(indent, ' ', /*ensure_ascii=*/false);
}

}  // namespace mineru

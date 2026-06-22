// Copyright (c) mlx-mineru.
// Output renderer: faithful C++ port of MinerU's union_make
// (mineru/backend/vlm/vlm_middle_json_mkcontent.py). Turns a middle_json
// `pdf_info` array into Markdown (mm/nlp) or content_list (v1/v2) JSON.
#pragma once

#include <string>

#include "nlohmann/json.hpp"

namespace mineru {

struct LatexDelimiters {
  std::string display_left = "$$";
  std::string display_right = "$$";
  std::string inline_left = "$";
  std::string inline_right = "$";
};

// `make_mode` is one of mineru::make_mode::{kMmMd,kNlpMd,kContentList,kContentListV2}.
// Returns a JSON string for the markdown modes, or a JSON array for content modes
// (mirroring Python union_make's str-or-list return).
nlohmann::json union_make(const nlohmann::json& pdf_info, const std::string& make_mode,
                          const std::string& img_bucket_path = "",
                          bool formula_enable = true, bool table_enable = true,
                          const LatexDelimiters& delim = {});

}  // namespace mineru

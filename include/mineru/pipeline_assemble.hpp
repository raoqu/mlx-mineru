// Copyright (c) mlx-mineru.
// Pipeline-backend assembly (P5): raw model output (layout_dets) -> middle_json page_info,
// faithful to MinerU MagicModel + model_json_to_middle_json + para_split. The resulting
// page_info feeds the existing (byte-exact) union_make renderer.
//
// Current coverage: the text/title path (text, paragraph_title, doc_title, headers/footers
// -> discarded). image/table/formula spans are a follow-up.
#pragma once

#include "nlohmann/json.hpp"

namespace mineru {

// model_page = {"layout_dets": [...], "page_info": {"width","height",...}}.
// page_w/page_h = PDF page size in points (middle_json page_size). Returns a page_info
// dict: {preproc_blocks, page_idx, page_size, discarded_blocks, para_blocks}.
nlohmann::json assemble_page_info(const nlohmann::json& model_page, int page_w, int page_h,
                                  int page_idx);

}  // namespace mineru

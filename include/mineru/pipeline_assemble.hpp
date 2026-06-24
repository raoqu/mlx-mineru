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

// optimize_formula_number_blocks: associate each formula_number block with an adjacent
// interline_equation, append \tag{N} (N = the number's OCR text), and drop the merged
// formula_number; unmatched formula_number blocks downgrade to text. Mutates `blocks` (a
// preproc_blocks/para_blocks array) in place. Run after post-OCR text-fill.
void optimize_formula_numbers(nlohmann::json& blocks);

// One embedded PDF character (digital-PDF text path); bbox in top-left page points.
struct PageChar {
  unsigned int cp = 0;
  int idx = 0;
  double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

// Digital-PDF text fill: assign embedded chars to text spans by bbox (calculate_char_in_span)
// and build span content (chars_to_content), faithful to MinerU txt_spans_extract. Returns
// the number of text spans left empty (candidates for OCR fallback). Mutates page_info.
int fill_chars_in_page(nlohmann::json& page_info, const std::vector<PageChar>& chars);

}  // namespace mineru

// Copyright (c) mlx-mineru.
// Post-OCR text fill (pipeline P5): the assembly leaves span text blank because the
// model_list ocr_text boxes carry no text; MinerU fills it by re-running OCR rec on each
// span's crop. This crops each span from the page image (bbox*scale, rotate-if-tall) and
// runs batched rec, faithful to model_json_to_middle_json._apply_post_ocr.
#pragma once

#include <cstdint>
#include <vector>

#include "mineru/ocr_rec.hpp"
#include "nlohmann/json.hpp"

namespace mineru {

// Fill span "content"/"score" in page_info's preproc_blocks + discarded_blocks in place.
// rgb = page image (w*h*3); scale = image_w / page_point_w (span bboxes are page-point).
void fill_span_text(nlohmann::json& page_info, const std::vector<uint8_t>& rgb, int w, int h,
                    double scale, const TextRecognizer& rec, float min_confidence = 0.5f);

}  // namespace mineru

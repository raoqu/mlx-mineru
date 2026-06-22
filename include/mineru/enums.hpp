// Copyright (c) mlx-mineru.
// Enum/string constants mirrored from MinerU `mineru/utils/enum_class.py` (v3.4.0).
// MinerU uses bare string values (not int enums); we keep them as string constants
// so the on-disk contract (middle_json / content_list) matches byte-for-byte.
#pragma once

namespace mineru {

// Project version of the data contract we target (source project version).
inline constexpr const char* kMineruVersion = "3.4.0";

// BlockType — paragraph/region level types.
namespace block_type {
inline constexpr const char* kImage = "image";
inline constexpr const char* kTable = "table";
inline constexpr const char* kChart = "chart";
inline constexpr const char* kImageBody = "image_body";
inline constexpr const char* kTableBody = "table_body";
inline constexpr const char* kChartBody = "chart_body";
inline constexpr const char* kCaption = "caption";
inline constexpr const char* kImageCaption = "image_caption";
inline constexpr const char* kTableCaption = "table_caption";
inline constexpr const char* kChartCaption = "chart_caption";
inline constexpr const char* kAlgorithmCaption = "algorithm_caption";
inline constexpr const char* kFootnote = "footnote";
inline constexpr const char* kImageFootnote = "image_footnote";
inline constexpr const char* kTableFootnote = "table_footnote";
inline constexpr const char* kChartFootnote = "chart_footnote";
inline constexpr const char* kText = "text";
inline constexpr const char* kTitle = "title";
inline constexpr const char* kInterlineEquation = "interline_equation";
inline constexpr const char* kEquation = "equation";
inline constexpr const char* kList = "list";
inline constexpr const char* kIndex = "index";
inline constexpr const char* kDiscarded = "discarded";
// vlm 2.5
inline constexpr const char* kCode = "code";
inline constexpr const char* kCodeBody = "code_body";
inline constexpr const char* kCodeCaption = "code_caption";
inline constexpr const char* kCodeFootnote = "code_footnote";
inline constexpr const char* kAlgorithm = "algorithm";
inline constexpr const char* kRefText = "ref_text";
inline constexpr const char* kPhonetic = "phonetic";
inline constexpr const char* kHeader = "header";
inline constexpr const char* kFooter = "footer";
inline constexpr const char* kPageNumber = "page_number";
inline constexpr const char* kAsideText = "aside_text";
inline constexpr const char* kPageFootnote = "page_footnote";
// pp_doclayout_v2
inline constexpr const char* kAbstract = "abstract";
inline constexpr const char* kDocTitle = "doc_title";
inline constexpr const char* kParagraphTitle = "paragraph_title";
inline constexpr const char* kVerticalText = "vertical_text";
inline constexpr const char* kHeaderImage = "header_image";
inline constexpr const char* kFooterImage = "footer_image";
inline constexpr const char* kFormulaNumber = "formula_number";
}  // namespace block_type

// ContentType — span level types.
namespace content_type {
inline constexpr const char* kImage = "image";
inline constexpr const char* kTable = "table";
inline constexpr const char* kChart = "chart";
inline constexpr const char* kText = "text";
inline constexpr const char* kInterlineEquation = "interline_equation";
inline constexpr const char* kInlineEquation = "inline_equation";
inline constexpr const char* kEquation = "equation";
inline constexpr const char* kHyperlink = "hyperlink";
}  // namespace content_type

// MakeMode — output renderer selector for union_make.
namespace make_mode {
inline constexpr const char* kMmMd = "mm_markdown";
inline constexpr const char* kNlpMd = "nlp_markdown";
inline constexpr const char* kContentList = "content_list";
inline constexpr const char* kContentListV2 = "content_list_v2";
}  // namespace make_mode

// ModelPath — model repo ids and on-disk subpaths.
namespace model_path {
inline constexpr const char* kVlmRootHf = "opendatalab/MinerU2.5-Pro-2605-1.2B";
inline constexpr const char* kVlmRootModelScope = "OpenDataLab/MinerU2.5-Pro-2605-1.2B";
inline constexpr const char* kPipelineRootHf = "opendatalab/PDF-Extract-Kit-1.0";
inline constexpr const char* kPipelineRootModelScope = "OpenDataLab/PDF-Extract-Kit-1.0";
inline constexpr const char* kPpDocLayoutV2 = "models/Layout/PP-DocLayoutV2";
inline constexpr const char* kUnimernetSmall = "models/MFR/unimernet_hf_small_2503";
inline constexpr const char* kPpFormulaNetPlusM = "models/MFR/pp_formulanet_plus_m";
inline constexpr const char* kPytorchPaddle = "models/OCR/paddleocr_torch";
inline constexpr const char* kSlanetPlus = "models/TabRec/SlanetPlus/slanet-plus.onnx";
inline constexpr const char* kUnetStructure = "models/TabRec/UnetStructure/unet.onnx";
inline constexpr const char* kPaddleTableCls =
    "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx";
}  // namespace model_path

}  // namespace mineru

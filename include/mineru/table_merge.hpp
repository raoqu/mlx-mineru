// Copyright (c) mlx-mineru.
// Cross-page table merge (pipeline backend): faithful port of MinerU's
// mineru/utils/table_merge.py merge_table(). Merges a table at the top of page N into a
// table at the bottom of page N-1 when their structure (columns / repeated header / width)
// indicates a continuation. Mutates pdf_info in place: the previous table's body HTML grows,
// carried footnotes get `cross_page`+reindex, and every sub-block of the merged-away table is
// emptied (lines=[], lines_deleted=true).
#pragma once

#include "nlohmann/json.hpp"

namespace mineru {

// Gated by MINERU_TABLE_MERGE_ENABLE (default on), matching cross_page_table_merge().
void cross_page_table_merge(nlohmann::json& pdf_info);

}  // namespace mineru

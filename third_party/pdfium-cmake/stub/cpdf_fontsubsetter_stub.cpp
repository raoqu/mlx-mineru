// Stub for CPDF_FontSubsetter, used in place of the real implementation (which depends on
// harfbuzz's subsetter, hb-subset.h). The subsetter only produces override objects for
// *newly embedded* fonts during save; mlx-mineru's only save path (draw_bbox) adds overlays
// using the standard 14 fonts (e.g. Helvetica), which are never embedded/subsetted — so the
// real subsetter would return no overrides here too. Returning an empty map is therefore
// behaviour-preserving for this build and lets us drop the harfbuzz dependency entirely.
#include "core/fpdfapi/edit/cpdf_fontsubsetter.h"

#include "core/fpdfapi/parser/cpdf_dictionary.h"
#include "core/fpdfapi/parser/cpdf_object.h"
#include "core/fpdfapi/parser/cpdf_stream.h"

CPDF_FontSubsetter::CPDF_FontSubsetter(CPDF_Document* doc) : doc_(doc) {}

CPDF_FontSubsetter::~CPDF_FontSubsetter() = default;

// The `candidates_` map member makes the (defaulted) outer dtor require these.
CPDF_FontSubsetter::SubsetCandidate::SubsetCandidate() = default;

CPDF_FontSubsetter::SubsetCandidate::~SubsetCandidate() = default;

std::map<uint32_t, RetainPtr<const CPDF_Object>>
CPDF_FontSubsetter::GenerateObjectOverrides(pdfium::span<const uint32_t>) {
  return {};
}

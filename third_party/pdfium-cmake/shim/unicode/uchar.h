// Minimal ICU shim for pdfium's core/fxcrt/fx_extension.h, which (with USE_SYSTEM_ICUUC)
// includes <unicode/uchar.h> but uses only these 7 character-classification helpers. Backing
// them with <wctype.h> lets the trimmed static build avoid an ICU dependency entirely (no
// libicucore / no bundled ICU). Adequate for pdfium's tokenizing / text handling on the BMP.
#ifndef PDFIUM_TRIM_ICU_SHIM_UNICODE_UCHAR_H_
#define PDFIUM_TRIM_ICU_SHIM_UNICODE_UCHAR_H_

#include <stdint.h>
#include <wctype.h>

typedef int32_t UChar32;

static inline int8_t u_isalnum(UChar32 c) { return iswalnum((wint_t)c) ? 1 : 0; }
static inline int8_t u_isalpha(UChar32 c) { return iswalpha((wint_t)c) ? 1 : 0; }
static inline int8_t u_islower(UChar32 c) { return iswlower((wint_t)c) ? 1 : 0; }
static inline int8_t u_isspace(UChar32 c) { return iswspace((wint_t)c) ? 1 : 0; }
static inline int8_t u_isupper(UChar32 c) { return iswupper((wint_t)c) ? 1 : 0; }
static inline UChar32 u_tolower(UChar32 c) { return (UChar32)towlower((wint_t)c); }
static inline UChar32 u_toupper(UChar32 c) { return (UChar32)towupper((wint_t)c); }

#endif  // PDFIUM_TRIM_ICU_SHIM_UNICODE_UCHAR_H_

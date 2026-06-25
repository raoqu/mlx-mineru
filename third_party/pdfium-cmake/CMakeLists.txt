cmake_minimum_required(VERSION 3.20)
project(pdfium_static C CXX)

# Build a *static* libpdfium.a from a trimmed pdfium source tree (PDFIUM_SRC), macOS/arm64
# only, with no gn/depot_tools. Trimmed: no V8/JS, no XFA, no Skia, no partition_alloc, no
# abseil (patched out), no png/tiff/bmp/gif codecs (progressive_decoder excluded), system
# zlib. Only the components reachable from the ~38 FPDF_* APIs mlx-mineru calls are built.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_OSX_ARCHITECTURES arm64)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-w -fno-exceptions -fno-rtti)          # pdfium builds without exceptions/RTTI

if(NOT DEFINED PDFIUM_SRC)
  message(FATAL_ERROR "pass -DPDFIUM_SRC=<path to .pdfium-src>")
endif()
set(P ${PDFIUM_SRC})
set(TP ${P}/third_party)

# ---------------------------------------------------------------------------
# Helper: recursive glob with exclusion of tests / other-platform / disabled features.
# ---------------------------------------------------------------------------
function(pdfium_sources outvar)
  set(acc "")
  foreach(dir ${ARGN})
    file(GLOB_RECURSE g CONFIGURE_DEPENDS ${dir}/*.cpp ${dir}/*.cc ${dir}/*.c)
    list(APPEND acc ${g})
  endforeach()
  list(FILTER acc EXCLUDE REGEX "_(unit|embedder)?test\\.(cpp|cc)$")
  list(FILTER acc EXCLUDE REGEX "_test\\.(cpp|cc)$")
  list(FILTER acc EXCLUDE REGEX "/(xfa|skia|skrifa|win32|win|android|linux)/")
  list(FILTER acc EXCLUDE REGEX "_(win|windows|linux|android)\\.(cpp|cc)$")
  list(FILTER acc EXCLUDE REGEX "windowsrenderdevice")          # cfx_ + cpdf_ win render device
  list(FILTER acc EXCLUDE REGEX "fx_memory_pa\\.cpp$")          # partition_alloc allocator
  list(FILTER acc EXCLUDE REGEX "progressive_decoder\\.cpp$")   # pulls png/tiff/bmp/gif
  list(FILTER acc EXCLUDE REGEX "fpdf_javascript\\.cpp$")       # V8
  list(FILTER acc EXCLUDE REGEX "cpdf_fontsubsetter\\.cpp$")    # harfbuzz subsetter (not needed)
  list(FILTER acc EXCLUDE REGEX "/testing/")                    # test support
  list(FILTER acc EXCLUDE REGEX "/test_[a-z_]+\\.(cpp|cc)$")    # test_with_page_module etc.
  list(FILTER acc EXCLUDE REGEX "code_point_view\\.cpp$")       # win-only (16-bit wchar_t); .h is inline on mac
  set(${outvar} ${acc} PARENT_SCOPE)
endfunction()

# ---- pdfium core + fpdfsdk (the C API) ----
pdfium_sources(CORE
  ${P}/core/fxcrt
  ${P}/core/fxge/agg ${P}/core/fxge/apple ${P}/core/fxge/dib
  ${P}/core/fxge/fontdata ${P}/core/fxge/freetype ${P}/core/fxge/text
  ${P}/core/fpdfapi ${P}/core/fpdftext ${P}/core/fpdfdoc ${P}/core/fdrm
  ${P}/constants
)
# fxge top-level files (not in a subdir)
file(GLOB FXGE_TOP ${P}/core/fxge/*.cpp)
list(FILTER FXGE_TOP EXCLUDE REGEX "test|cfx_windows")
# fxcodec: only the codecs reachable from PDF content streams
file(GLOB FXCODEC_TOP ${P}/core/fxcodec/*.cpp)
list(FILTER FXCODEC_TOP EXCLUDE REGEX "test|progressive_decoder")
pdfium_sources(FXCODEC
  ${P}/core/fxcodec/basic ${P}/core/fxcodec/fax ${P}/core/fxcodec/flate
  ${P}/core/fxcodec/jbig2 ${P}/core/fxcodec/jpeg ${P}/core/fxcodec/jpx
  ${P}/core/fxcodec/icc
)
# fpdfsdk C API + form filling + widget list (FPDF_FFLDraw / forms)
file(GLOB FPDFSDK_TOP ${P}/fpdfsdk/*.cpp)
list(FILTER FPDFSDK_TOP EXCLUDE REGEX "test|fpdf_javascript")
pdfium_sources(FPDFSDK_SUB ${P}/fpdfsdk/formfiller ${P}/fpdfsdk/pwl)

# fxjs: the no-V8 stub runtime (IJS_Runtime::Create returns a do-nothing JS engine), so forms
# work without a JavaScript engine.
set(FXJS ${P}/fxjs/cjs_runtimestub.cpp ${P}/fxjs/cjs_event_context_stub.cpp ${P}/fxjs/ijs_runtime.cpp)
# Replace the harfbuzz-dependent font subsetter with a behaviour-preserving stub.
set(STUBS ${CMAKE_CURRENT_SOURCE_DIR}/stub/cpdf_fontsubsetter_stub.cpp)

# ---- in-tree third-party: agg (rasterizer), lcms (icc), libopenjpeg (jpx) ----
file(GLOB AGG ${TP}/agg23/*.cpp)
file(GLOB LCMS ${TP}/lcms/src/*.c)
file(GLOB OPENJPEG ${TP}/libopenjpeg/*.c ${TP}/libopenjpeg/*.cc)
list(FILTER OPENJPEG EXCLUDE REGEX "t1_generate_luts")   # host LUT generator (has its own main)

# ---- freetype (font rasterizer) — module amalgamation files only ----
set(FT ${TP}/freetype/src/src)
set(FREETYPE
  ${FT}/autofit/autofit.c ${FT}/base/ftbase.c ${FT}/base/ftbbox.c
  ${FT}/base/ftbitmap.c ${FT}/base/ftfstype.c ${FT}/base/ftgasp.c
  ${FT}/base/ftglyph.c ${FT}/base/ftinit.c ${FT}/base/ftmm.c
  ${FT}/base/ftstroke.c ${FT}/base/ftsynth.c ${FT}/base/ftsystem.c
  ${FT}/base/fttype1.c ${FT}/cff/cff.c ${FT}/cid/type1cid.c
  ${FT}/psaux/psaux.c ${FT}/pshinter/pshinter.c ${FT}/psnames/psnames.c
  ${FT}/raster/raster.c ${FT}/sfnt/sfnt.c ${FT}/smooth/smooth.c
  ${FT}/truetype/truetype.c ${FT}/type1/type1.c ${FT}/gzip/ftgzip.c
  ${FT}/base/ftdebug.c                                  # FT_Trace_Enable/Disable
)

# ---- libjpeg-turbo (jpeg decode), C-only (no SIMD); multi-precision (8/12/16-bit) ----
# Exact source sets from chromium libjpeg_turbo BUILD.gn. The same files are compiled at
# three sample precisions (BITS_IN_JSAMPLE 8/12/16), each producing precision-tagged symbols
# (jinit_* / j12init_* / j16init_*) that jdmaster dispatches by data_precision.
set(JT ${TP}/libjpeg_turbo/src)
set(_JPEG16 jcapistd jccolor jcdiffct jclossls jcmainct jcprepct jcsample
            jdapistd jdcolor jddiffct jdlossls jdmainct jdpostct jdsample jutils)
set(_JPEG12_EXTRA jccoefct jcdctmgr jdcoefct jddctmgr jdmerge jfdctfst jfdctint
                  jidctflt jidctfst jidctint jidctred jquant1 jquant2)
set(_JPEG8_EXTRA jcapimin jchuff jcicc jcinit jclhuff jcmarker jcmaster jcomapi
                 jcparam jcphuff jctrans jdapimin jdatadst jdatasrc jdhuff jdicc
                 jdinput jdlhuff jdmarker jdmaster jdphuff jdtrans jerror jfdctflt
                 jmemmgr jmemnobs jpeg_nbits)
set(LIBJPEG16 "")
set(LIBJPEG12 "")
set(LIBJPEG8 "")
foreach(f ${_JPEG16})
  list(APPEND LIBJPEG16 ${JT}/${f}.c)
endforeach()
foreach(f ${_JPEG16} ${_JPEG12_EXTRA})
  list(APPEND LIBJPEG12 ${JT}/${f}.c)
endforeach()
foreach(f ${_JPEG16} ${_JPEG12_EXTRA} ${_JPEG8_EXTRA})
  list(APPEND LIBJPEG8 ${JT}/${f}.c)
endforeach()
# 12/16-bit variants as object libraries (same files, different precision define -> tagged symbols)
add_library(jpeg16 OBJECT ${LIBJPEG16})
target_include_directories(jpeg16 PRIVATE ${JT})
target_compile_definitions(jpeg16 PRIVATE BITS_IN_JSAMPLE=16)
target_compile_options(jpeg16 PRIVATE -w)
add_library(jpeg12 OBJECT ${LIBJPEG12})
target_include_directories(jpeg12 PRIVATE ${JT})
target_compile_definitions(jpeg12 PRIVATE BITS_IN_JSAMPLE=12)
target_compile_options(jpeg12 PRIVATE -w)

add_library(pdfium STATIC
  ${CORE} ${FXGE_TOP} ${FXCODEC_TOP} ${FXCODEC} ${FPDFSDK_TOP} ${FPDFSDK_SUB}
  ${FXJS} ${STUBS}
  ${AGG} ${LCMS} ${OPENJPEG} ${FREETYPE} ${LIBJPEG8}
  $<TARGET_OBJECTS:jpeg12> $<TARGET_OBJECTS:jpeg16>
)

target_include_directories(pdfium PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/shim       # <unicode/uchar.h> ICU shim (USE_SYSTEM_ICUUC)
  ${P}                                   # core/ public/ constants/ fpdfsdk/ build/ third_party/
  ${TP}/freetype/include                 # freetype-custom-config/{ftoption,ftmodule}.h
  ${TP}/freetype/src/include             # freetype public API
  ${JT}                                  # jpeglib.h + jconfig.h (chromium libjpeg_turbo/src)
  ${TP}/libopenjpeg
  ${TP}/lcms/include
)

target_compile_definitions(pdfium PRIVATE
  USE_SYSTEM_ZLIB=1                       # flate -> system <zlib.h>
  USE_SYSTEM_ICUUC=1                      # fx_extension.h -> <unicode/uchar.h> (our wctype shim)
  PDF_USE_AGG=1                           # AGG rasterizer backend (not Skia)
  # freetype: build as a library with pdfium's custom config
  FT2_BUILD_LIBRARY
  FT_CONFIG_MODULES_H=<freetype-custom-config/ftmodule.h>
  FT_CONFIG_OPTIONS_H=<freetype-custom-config/ftoption.h>
  # libjpeg-turbo C build (no SIMD)
  # NOT defined: PDF_USE_PARTITION_ALLOC, PDF_ENABLE_V8, PDF_ENABLE_XFA, _SKIA_SUPPORT_
)
# lcms needs CMS_DLL off / static; openjpeg static config:
target_compile_definitions(pdfium PRIVATE OPJ_STATIC CMS_NO_REGISTER_KEYWORD=1)

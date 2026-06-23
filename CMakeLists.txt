cmake_minimum_required(VERSION 3.20)
project(mlx_mineru LANGUAGES CXX)

# ---- Standard / flags --------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ---- Vendored header-only dependencies --------------------------------------
add_library(thirdparty_headers INTERFACE)
target_include_directories(thirdparty_headers INTERFACE
  ${CMAKE_SOURCE_DIR}/third_party        # nlohmann/json.hpp, CLI11/CLI11.hpp
)

# ---- pdfium (prebuilt; fetch via scripts/fetch_pdfium.sh) --------------------
set(PDFIUM_DIR ${CMAKE_SOURCE_DIR}/third_party/pdfium)
if(EXISTS ${PDFIUM_DIR}/lib/libpdfium.dylib)
  add_library(pdfium SHARED IMPORTED)
  set_target_properties(pdfium PROPERTIES
    IMPORTED_LOCATION ${PDFIUM_DIR}/lib/libpdfium.dylib
    INTERFACE_INCLUDE_DIRECTORIES ${PDFIUM_DIR}/include)
  set(HAVE_PDFIUM TRUE)
else()
  message(WARNING "pdfium not found at ${PDFIUM_DIR}; run scripts/fetch_pdfium.sh. PDF targets disabled.")
  set(HAVE_PDFIUM FALSE)
endif()

# ---- MLX (C++ library bundled in the pip `mlx` package) ----------------------
execute_process(
  COMMAND python3 -c "import mlx.core,os;print(os.path.dirname(mlx.core.__file__))"
  OUTPUT_VARIABLE MLX_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE MLX_RC)
if(MLX_RC EQUAL 0 AND EXISTS ${MLX_ROOT}/share/cmake/MLX/MLXConfig.cmake)
  find_package(MLX CONFIG PATHS ${MLX_ROOT}/share/cmake/MLX NO_DEFAULT_PATH QUIET)
endif()
if(MLX_FOUND)
  set(HAVE_MLX TRUE)
  set(MLX_LIB_DIR ${MLX_ROOT}/lib)
  message(STATUS "Found MLX at ${MLX_ROOT}")
else()
  message(WARNING "MLX not found (pip `mlx`); MLX targets disabled.")
  set(HAVE_MLX FALSE)
endif()

# ---- Core library ------------------------------------------------------------
add_library(mineru_core STATIC
  src/core/middle_json.cpp
  src/output/text_utils.cpp
  src/output/mkcontent.cpp
)
target_include_directories(mineru_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(mineru_core PRIVATE
  ${CMAKE_SOURCE_DIR}/src/output
)
target_link_libraries(mineru_core PUBLIC thirdparty_headers)

# ---- Tests -------------------------------------------------------------------
enable_testing()

add_executable(test_middle_json_roundtrip tests/test_middle_json_roundtrip.cpp)
target_include_directories(test_middle_json_roundtrip PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_middle_json_roundtrip PRIVATE mineru_core)
add_test(
  NAME middle_json_roundtrip
  COMMAND test_middle_json_roundtrip
          ${CMAKE_SOURCE_DIR}/tests/data/sample_middle.json
)

add_executable(test_mkcontent_golden tests/test_mkcontent_golden.cpp)
target_include_directories(test_mkcontent_golden PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_mkcontent_golden PRIVATE mineru_core)
add_test(
  NAME mkcontent_golden
  COMMAND test_mkcontent_golden ${CMAKE_SOURCE_DIR}/tests
)

# ---- PDF rasterization (pdfium) ---------------------------------------------
if(HAVE_PDFIUM)
  add_library(mineru_pdf STATIC src/io/pdf.cpp)
  target_include_directories(mineru_pdf PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(mineru_pdf PUBLIC pdfium thirdparty_headers)

  add_executable(test_pdf_raster tests/test_pdf_raster.cpp)
  target_include_directories(test_pdf_raster PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_pdf_raster PRIVATE mineru_pdf)
  set_target_properties(test_pdf_raster PROPERTIES
    BUILD_RPATH ${PDFIUM_DIR}/lib)
  add_test(
    NAME pdf_raster
    COMMAND test_pdf_raster ${CMAKE_SOURCE_DIR}/tests/golden/pdf_raster.json
            $ENV{HOME}/research/MinerU/demo/pdfs
  )
endif()

# ---- VLM support lib (tokenizer; grows into the Qwen2-VL backend) ------------
add_library(mineru_vlm STATIC
  src/vlm/tokenizer.cpp
)
target_include_directories(mineru_vlm
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src/vlm)
target_link_libraries(mineru_vlm PUBLIC thirdparty_headers)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_tokenizer PRIVATE mineru_vlm)
add_test(NAME tokenizer
  COMMAND test_tokenizer ${CMAKE_SOURCE_DIR}/models/MinerU2.5-tokenizer
          ${CMAKE_SOURCE_DIR}/tests/golden/tokenizer.json)

# ---- MLX smoke test ---------------------------------------------------------
if(HAVE_MLX)
  add_executable(test_mlx_smoke tests/test_mlx_smoke.cpp)
  target_include_directories(test_mlx_smoke PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_mlx_smoke PRIVATE mlx thirdparty_headers)
  set_target_properties(test_mlx_smoke PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
  add_test(NAME mlx_smoke COMMAND test_mlx_smoke)
endif()

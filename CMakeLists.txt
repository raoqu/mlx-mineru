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

  # Prefer the vendored MLX runtime (scripts/fetch_mlx.sh) so the binary loads
  # libmlx.dylib + mlx.metallib from the repo, not from a Python site-packages
  # tree. We keep the official MLXConfig for the framework/flag interface but
  # repoint the dylib location (-> rpath) and headers to third_party/mlx/.
  set(VENDORED_MLX ${CMAKE_SOURCE_DIR}/third_party/mlx)
  if(EXISTS ${VENDORED_MLX}/lib/libmlx.dylib)
    # Override the generic AND per-config imported locations so linking and the
    # auto-generated rpath both point at the vendored dylib (not site-packages).
    set_target_properties(mlx PROPERTIES
      IMPORTED_LOCATION ${VENDORED_MLX}/lib/libmlx.dylib
      IMPORTED_LOCATION_RELEASE ${VENDORED_MLX}/lib/libmlx.dylib
      IMPORTED_LOCATION_DEBUG ${VENDORED_MLX}/lib/libmlx.dylib
      IMPORTED_LOCATION_RELWITHDEBINFO ${VENDORED_MLX}/lib/libmlx.dylib
      IMPORTED_LOCATION_MINSIZEREL ${VENDORED_MLX}/lib/libmlx.dylib)
    if(EXISTS ${VENDORED_MLX}/include/mlx/mlx.h)
      set_target_properties(mlx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${VENDORED_MLX}/include)
    endif()
    set(MLX_LIB_DIR ${VENDORED_MLX}/lib)
    message(STATUS "Runtime MLX vendored at ${MLX_LIB_DIR} (decoupled from site-packages)")
  endif()
else()
  message(WARNING "MLX not found (pip `mlx`); MLX targets disabled.")
  set(HAVE_MLX FALSE)
endif()

# ---- ONNX Runtime (C++; pipeline backend) -----------------------------------
set(ORT_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
if(EXISTS ${ORT_DIR}/lib/libonnxruntime.dylib)
  add_library(onnxruntime SHARED IMPORTED)
  file(GLOB _ort_dylib ${ORT_DIR}/lib/libonnxruntime.*.dylib)
  list(GET _ort_dylib 0 _ort_real)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${_ort_real}
    INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/include)
  set(HAVE_ORT TRUE)
  message(STATUS "Found ONNX Runtime at ${ORT_DIR}")
else()
  message(WARNING "ONNX Runtime not found; run scripts/fetch_onnxruntime.sh. Pipeline targets disabled.")
  set(HAVE_ORT FALSE)
endif()

# ---- Core library ------------------------------------------------------------
add_library(mineru_core STATIC
  src/core/middle_json.cpp
  src/core/pipeline_assemble.cpp
  src/output/text_utils.cpp
  src/output/mkcontent.cpp
  src/io/image_write.cpp
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

add_executable(test_pipeline_assemble tests/test_pipeline_assemble.cpp)
target_include_directories(test_pipeline_assemble PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_pipeline_assemble PRIVATE mineru_core)
add_test(
  NAME pipeline_assemble
  COMMAND test_pipeline_assemble ${CMAKE_SOURCE_DIR}/tests/golden
)

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/demo1_p2_model.json)
  add_executable(test_pipeline_formula tests/test_pipeline_formula.cpp)
  target_include_directories(test_pipeline_formula PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_pipeline_formula PRIVATE mineru_core)
  add_test(NAME pipeline_formula COMMAND test_pipeline_formula ${CMAKE_SOURCE_DIR}/tests/golden)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/demo1_p5_model.json)
  add_executable(test_pipeline_table tests/test_pipeline_table.cpp)
  target_include_directories(test_pipeline_table PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_pipeline_table PRIVATE mineru_core)
  add_test(NAME pipeline_table COMMAND test_pipeline_table ${CMAKE_SOURCE_DIR}/tests/golden)
endif()

add_executable(test_formula_number tests/test_formula_number.cpp)
target_include_directories(test_formula_number PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_formula_number PRIVATE mineru_core)
add_test(NAME formula_number COMMAND test_formula_number)

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

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/pipeline_p0_model.json)
    add_executable(test_pipeline_digital tests/test_pipeline_digital.cpp)
    target_include_directories(test_pipeline_digital PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_digital PRIVATE mineru_pdf mineru_core)
    set_target_properties(test_pipeline_digital PROPERTIES BUILD_RPATH ${PDFIUM_DIR}/lib)
    add_test(NAME pipeline_digital COMMAND test_pipeline_digital
              ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/a.pdf)
  endif()

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
  src/vlm/image_preprocess.cpp
  src/vlm/vlm_layout.cpp
  src/vlm/otsl.cpp
  src/vlm/post_process.cpp
)
target_include_directories(mineru_vlm
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src/vlm)
target_link_libraries(mineru_vlm PUBLIC thirdparty_headers)

# ---- Pipeline backend library (defined early so the CLI can link it) --------
if(HAVE_ORT)
  add_library(mineru_pipeline STATIC src/pipeline/layout_det.cpp src/pipeline/table_cls.cpp src/pipeline/ocr_rec.cpp src/pipeline/ocr_det.cpp src/pipeline/ocr.cpp src/pipeline/formula_rec.cpp src/pipeline/table_rec.cpp src/pipeline/post_ocr.cpp src/pipeline/pipeline_driver.cpp src/pipeline/wired_table.cpp)
  target_include_directories(mineru_pipeline PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(mineru_pipeline PUBLIC onnxruntime mineru_vlm mineru_core thirdparty_headers)
endif()

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_tokenizer PRIVATE mineru_vlm)
add_test(NAME tokenizer
  COMMAND test_tokenizer ${CMAKE_SOURCE_DIR}/models/MinerU2.5-tokenizer
          ${CMAKE_SOURCE_DIR}/tests/golden/tokenizer.json)

add_executable(test_preprocess tests/test_preprocess.cpp)
target_include_directories(test_preprocess PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_preprocess PRIVATE mineru_vlm)
add_test(NAME preprocess COMMAND test_preprocess ${CMAKE_SOURCE_DIR}/tests/golden)

add_executable(test_layout tests/test_layout.cpp)
target_include_directories(test_layout PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_layout PRIVATE mineru_vlm)
add_test(NAME layout COMMAND test_layout ${CMAKE_SOURCE_DIR}/tests/golden/layout.json)

add_executable(test_otsl tests/test_otsl.cpp)
target_include_directories(test_otsl PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_otsl PRIVATE mineru_vlm)
add_test(NAME otsl COMMAND test_otsl ${CMAKE_SOURCE_DIR}/tests/golden/otsl.json)

add_executable(test_post_process tests/test_post_process.cpp)
target_include_directories(test_post_process PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_post_process PRIVATE mineru_vlm)
add_test(NAME post_process COMMAND test_post_process ${CMAKE_SOURCE_DIR}/tests/golden/post_process.json)

# ---- MLX smoke test ---------------------------------------------------------
if(HAVE_MLX)
  add_executable(test_mlx_smoke tests/test_mlx_smoke.cpp)
  target_include_directories(test_mlx_smoke PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_mlx_smoke PRIVATE mlx thirdparty_headers)
  set_target_properties(test_mlx_smoke PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
  add_test(NAME mlx_smoke COMMAND test_mlx_smoke)

  # ---- Qwen2-VL model (MLX) -------------------------------------------------
  add_library(mineru_qwen STATIC src/vlm/qwen2_llm.cpp)
  target_include_directories(mineru_qwen PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(mineru_qwen PUBLIC mlx thirdparty_headers)

  # ---- mlx-mineru CLI -------------------------------------------------------
  if(HAVE_PDFIUM)
    add_executable(mlx-mineru src/cli/main.cpp)
    target_link_libraries(mlx-mineru PRIVATE mineru_qwen mineru_vlm mineru_pdf mineru_core)
    set(MLX_MINERU_RPATH "${MLX_LIB_DIR};${PDFIUM_DIR}/lib")
    if(HAVE_ORT)
      # Add the native pipeline backend (--backend pipeline) to the CLI.
      target_link_libraries(mlx-mineru PRIVATE mineru_pipeline)
      target_compile_definitions(mlx-mineru PRIVATE MINERU_HAVE_PIPELINE=1)
      set(MLX_MINERU_RPATH "${MLX_MINERU_RPATH};${ORT_DIR}/lib")
    endif()
    set_target_properties(mlx-mineru PROPERTIES BUILD_RPATH "${MLX_MINERU_RPATH}")
  endif()

  set(MODEL_ST ${CMAKE_SOURCE_DIR}/models/MinerU2.5-tokenizer/model.safetensors)
  if(EXISTS ${MODEL_ST})
    add_executable(test_llm_forward tests/test_llm_forward.cpp)
    target_include_directories(test_llm_forward PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_llm_forward PRIVATE mineru_qwen)
    set_target_properties(test_llm_forward PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
    add_test(NAME llm_forward
      COMMAND test_llm_forward ${MODEL_ST} ${CMAKE_SOURCE_DIR}/tests/golden/llm.json)

    add_executable(test_vision tests/test_vision.cpp)
    target_include_directories(test_vision PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_vision PRIVATE mineru_qwen mineru_vlm)
    set_target_properties(test_vision PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
    add_test(NAME vision COMMAND test_vision ${MODEL_ST} ${CMAKE_SOURCE_DIR}/tests/golden)

    add_executable(test_vlm tests/test_vlm.cpp)
    target_include_directories(test_vlm PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_vlm PRIVATE mineru_qwen mineru_vlm)
    set_target_properties(test_vlm PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
    add_test(NAME vlm COMMAND test_vlm ${MODEL_ST} ${CMAKE_SOURCE_DIR}/tests/golden)

    add_executable(test_vlm_batch tests/test_vlm_batch.cpp)
    target_include_directories(test_vlm_batch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_vlm_batch PRIVATE mineru_qwen mineru_vlm)
    set_target_properties(test_vlm_batch PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
    add_test(NAME vlm_batch COMMAND test_vlm_batch ${MODEL_ST} ${CMAKE_SOURCE_DIR}/tests/golden)
  else()
    message(STATUS "model weights not present; llm_forward test disabled (run scripts/fetch_weights.sh)")
  endif()
endif()

# ---- ONNX Runtime smoke test (pipeline) -------------------------------------
if(HAVE_ORT AND EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/Layout/layout.onnx)
  add_executable(test_onnx_smoke tests/test_onnx_smoke.cpp)
  target_include_directories(test_onnx_smoke PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_onnx_smoke PRIVATE onnxruntime)
  set_target_properties(test_onnx_smoke PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
  add_test(NAME onnx_smoke
    COMMAND test_onnx_smoke ${CMAKE_SOURCE_DIR}/models/pipeline/Layout/layout.onnx)
endif()

# ---- Pipeline backend tests (library defined earlier) -----------------------
if(HAVE_ORT)
  if(EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/Layout/layout.onnx)
    add_executable(test_layout_det tests/test_layout_det.cpp)
    target_include_directories(test_layout_det PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_layout_det PRIVATE mineru_pipeline)
    set_target_properties(test_layout_det PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME layout_det
      COMMAND test_layout_det ${CMAKE_SOURCE_DIR}/models/pipeline/Layout
              ${CMAKE_SOURCE_DIR}/tests/golden)

    if(HAVE_PDFIUM AND EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/pipeline_p0_model.json)
      add_executable(test_layout_order tests/test_layout_order.cpp)
      target_include_directories(test_layout_order PRIVATE ${CMAKE_SOURCE_DIR}/tests)
      target_link_libraries(test_layout_order PRIVATE mineru_pipeline mineru_pdf)
      set_target_properties(test_layout_order PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
      add_test(NAME layout_order COMMAND test_layout_order ${CMAKE_SOURCE_DIR}/models/pipeline/Layout
                ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/a.pdf)
    endif()
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx)
    add_executable(test_tab_cls tests/test_tab_cls.cpp)
    target_include_directories(test_tab_cls PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_tab_cls PRIVATE mineru_pipeline)
    set_target_properties(test_tab_cls PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME tab_cls
      COMMAND test_tab_cls ${CMAKE_SOURCE_DIR}/models/pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx
              ${CMAKE_SOURCE_DIR}/tests/golden)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx)
    add_executable(test_ocr_rec tests/test_ocr_rec.cpp)
    target_include_directories(test_ocr_rec PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_ocr_rec PRIVATE mineru_pipeline)
    set_target_properties(test_ocr_rec PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME ocr_rec
      COMMAND test_ocr_rec ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/tests/golden)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/ocr_det.json)
    add_executable(test_ocr_det tests/test_ocr_det.cpp)
    target_include_directories(test_ocr_det PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_ocr_det PRIVATE mineru_pipeline)
    set_target_properties(test_ocr_det PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME ocr_det COMMAND test_ocr_det ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_det.onnx)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/ocr_page.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_det.onnx AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx)
    add_executable(test_ocr_page tests/test_ocr_page.cpp)
    target_include_directories(test_ocr_page PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_ocr_page PRIVATE mineru_pipeline)
    set_target_properties(test_ocr_page PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME ocr_page COMMAND test_ocr_page ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_det.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ppocrv6_dict.txt)
  endif()

  if(HAVE_PDFIUM AND EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/pipeline_p0_model.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx)
    add_executable(test_pipeline_textfill tests/test_pipeline_textfill.cpp)
    target_include_directories(test_pipeline_textfill PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_textfill PRIVATE mineru_pipeline mineru_core mineru_pdf)
    set_target_properties(test_pipeline_textfill PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
    add_test(NAME pipeline_textfill COMMAND test_pipeline_textfill ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/a.pdf)

    add_executable(test_pipeline_md tests/test_pipeline_md.cpp)
    target_include_directories(test_pipeline_md PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_md PRIVATE mineru_pipeline mineru_core mineru_pdf)
    set_target_properties(test_pipeline_md PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
    add_test(NAME pipeline_md COMMAND test_pipeline_md ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/a.pdf)

    add_executable(test_pipeline_multipage tests/test_pipeline_multipage.cpp)
    target_include_directories(test_pipeline_multipage PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_multipage PRIVATE mineru_pipeline mineru_core mineru_pdf)
    set_target_properties(test_pipeline_multipage PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
    add_test(NAME pipeline_multipage COMMAND test_pipeline_multipage ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/a.pdf)

    if(EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/Layout/layout.onnx AND
       EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/OCR/ocr_det.onnx)
      add_executable(test_pipeline_scratch tests/test_pipeline_scratch.cpp)
      target_include_directories(test_pipeline_scratch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
      target_link_libraries(test_pipeline_scratch PRIVATE mineru_pipeline mineru_core mineru_pdf)
      set_target_properties(test_pipeline_scratch PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
      add_test(NAME pipeline_scratch COMMAND test_pipeline_scratch ${CMAKE_SOURCE_DIR}/tests/golden
                ${CMAKE_SOURCE_DIR}/models/pipeline ${CMAKE_SOURCE_DIR}/a.pdf)

      if(EXISTS ${CMAKE_SOURCE_DIR}/tests/demo1.pdf AND
         EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/demo1_p2_model.json AND
         EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/MFR/mfr_encoder.onnx)
        add_executable(test_pipeline_formula_scratch tests/test_pipeline_formula_scratch.cpp)
        target_include_directories(test_pipeline_formula_scratch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        target_link_libraries(test_pipeline_formula_scratch PRIVATE mineru_pipeline mineru_core mineru_pdf)
        set_target_properties(test_pipeline_formula_scratch PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
        add_test(NAME pipeline_formula_scratch COMMAND test_pipeline_formula_scratch
                  ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/models/pipeline
                  ${CMAKE_SOURCE_DIR}/tests/demo1.pdf)
      endif()

      if(EXISTS ${CMAKE_SOURCE_DIR}/tests/demo1.pdf AND
         EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/demo1_p5_model.json AND
         EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/TabRec/SlanetPlus/slanet-plus.onnx)
        add_executable(test_pipeline_table_scratch tests/test_pipeline_table_scratch.cpp)
        target_include_directories(test_pipeline_table_scratch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        target_link_libraries(test_pipeline_table_scratch PRIVATE mineru_pipeline mineru_core mineru_pdf)
        set_target_properties(test_pipeline_table_scratch PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
        add_test(NAME pipeline_table_scratch COMMAND test_pipeline_table_scratch
                  ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/models/pipeline
                  ${CMAKE_SOURCE_DIR}/tests/demo1.pdf)
      endif()
    endif()
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/mfr.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/MFR/mfr_encoder.onnx AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/MFR/mfr_decoder.onnx)
    add_executable(test_mfr tests/test_mfr.cpp)
    target_include_directories(test_mfr PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_mfr PRIVATE mineru_pipeline)
    set_target_properties(test_mfr PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME mfr COMMAND test_mfr ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/MFR/mfr_encoder.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/MFR/mfr_decoder.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/MFR/mfr_vocab.txt)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/wired_table.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/TabRec/UnetStructure/unet.onnx)
    add_executable(test_wired_table tests/test_wired_table.cpp)
    target_include_directories(test_wired_table PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_wired_table PRIVATE mineru_pipeline)
    set_target_properties(test_wired_table PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME wired_table COMMAND test_wired_table ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/TabRec/UnetStructure/unet.onnx)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/table_struct.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/models/pipeline/TabRec/SlanetPlus/slanet-plus.onnx)
    add_executable(test_table tests/test_table.cpp)
    target_include_directories(test_table PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_table PRIVATE mineru_pipeline)
    set_target_properties(test_table PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME table COMMAND test_table ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/models/pipeline/TabRec/SlanetPlus/slanet-plus.onnx
              ${CMAKE_SOURCE_DIR}/models/pipeline/TabRec/SlanetPlus/table_structure_dict.txt)
  endif()
endif()

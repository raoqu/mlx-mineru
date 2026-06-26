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

# ---- getmodel (vendored: HF/ModelScope model downloader) --------------------
# Reuses github.com/raoqu/getmodel's DownloadModel — REST-API based, with automatic source
# selection (HF <-> ModelScope), byte-range resume, and retry. Links the system libcurl
# (/usr/lib/libcurl.4.dylib — a base macOS dylib, like libz), so it adds no external dep.
add_library(mineru_getmodel STATIC ${CMAKE_SOURCE_DIR}/third_party/getmodel/getmodel.cpp)
target_include_directories(mineru_getmodel PUBLIC ${CMAKE_SOURCE_DIR}/third_party/getmodel)
target_link_libraries(mineru_getmodel PUBLIC curl)
target_compile_options(mineru_getmodel PRIVATE -w)   # vendored upstream; silence its warnings

# ---- pdfium ------------------------------------------------------------------
# Preferred: a trimmed, macOS-only *static* libpdfium.a built from source
# (scripts/build_pdfium_static_src.sh) so the binary carries no libpdfium.dylib at runtime.
# Fallback: the bblanchon prebuilt dylib (scripts/fetch_pdfium.sh).
set(PDFIUM_STATIC_DIR ${CMAKE_SOURCE_DIR}/third_party/pdfium-static)
set(PDFIUM_DIR ${CMAKE_SOURCE_DIR}/third_party/pdfium)
set(PDFIUM_STATIC FALSE)
if(EXISTS ${PDFIUM_STATIC_DIR}/lib/libpdfium.a)
  add_library(pdfium STATIC IMPORTED)
  set_target_properties(pdfium PROPERTIES
    IMPORTED_LOCATION ${PDFIUM_STATIC_DIR}/lib/libpdfium.a
    INTERFACE_INCLUDE_DIRECTORIES ${PDFIUM_STATIC_DIR}/include)
  # pdfium's macOS backend (CoreGraphics/CoreText quartz) + flate via system zlib.
  target_link_libraries(pdfium INTERFACE
    "-framework CoreGraphics" "-framework CoreFoundation" "-framework CoreText"
    "-framework AppKit" z)
  set(HAVE_PDFIUM TRUE)
  set(PDFIUM_STATIC TRUE)
  message(STATUS "Using static pdfium at ${PDFIUM_STATIC_DIR}")
elseif(EXISTS ${PDFIUM_DIR}/lib/libpdfium.dylib)
  add_library(pdfium SHARED IMPORTED)
  set_target_properties(pdfium PROPERTIES
    IMPORTED_LOCATION ${PDFIUM_DIR}/lib/libpdfium.dylib
    INTERFACE_INCLUDE_DIRECTORIES ${PDFIUM_DIR}/include)
  set(HAVE_PDFIUM TRUE)
else()
  message(WARNING "pdfium not found; run scripts/build_pdfium_static_src.sh (static) "
                  "or scripts/fetch_pdfium.sh (dylib). PDF targets disabled.")
  set(HAVE_PDFIUM FALSE)
endif()

# ---- MLX (C++/Metal) ---------------------------------------------------------
# Preferred: a from-source *static* MLX (scripts/build_mlx_static.sh) so libmlx links
# statically and the binary carries no libmlx.dylib / libjaccl.dylib at runtime. The
# Metal shader archive (mlx.metallib) is a data asset, not a dylib — it is copied next to
# the executable (see the CLI target) and loaded at runtime.
set(MLX_STATIC_DIR ${CMAKE_SOURCE_DIR}/third_party/mlx-static)
set(MLX_STATIC FALSE)
if(EXISTS ${MLX_STATIC_DIR}/lib/libmlx.a)
  # Pin MLX_DIR to the static install: find_package reuses a cached MLX_DIR (e.g. a pip
  # path from an earlier configure) and would otherwise ignore PATHS, silently linking the
  # dynamic libmlx.dylib instead of libmlx.a.
  set(MLX_DIR ${MLX_STATIC_DIR}/share/cmake/MLX CACHE PATH "" FORCE)
  # MLX's static export references jaccl::jaccl but ships no jacclConfig.cmake, so
  # find_package(MLX) can't resolve it. Include the targets file directly to define it.
  if(EXISTS ${MLX_STATIC_DIR}/lib/cmake/jaccl/jacclTargets.cmake)
    include(${MLX_STATIC_DIR}/lib/cmake/jaccl/jacclTargets.cmake)
  endif()
  find_package(MLX CONFIG PATHS ${MLX_STATIC_DIR}/share/cmake/MLX NO_DEFAULT_PATH QUIET)
  if(MLX_FOUND)
    set(HAVE_MLX TRUE)
    set(MLX_STATIC TRUE)
    set(MLX_LIB_DIR ${MLX_STATIC_DIR}/lib)   # holds mlx.metallib (runtime asset)
    message(STATUS "Using static MLX at ${MLX_STATIC_DIR}")
  endif()
endif()

# Fallback: the dynamic MLX from the pip `mlx` package (vendored dylib via fetch_mlx.sh).
if(NOT MLX_FOUND)
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
endif()  # NOT MLX_FOUND (static-MLX fallback to dynamic pip MLX)

# With MLX linked statically there is no libmlx.dylib to sit beside; the Metal backend
# instead loads mlx.metallib from next to the *executable* (via dladdr on the binary).
# Colocate it in the build dir at configure time so every MLX binary (CLI + tests) finds it.
if(HAVE_MLX AND MLX_STATIC AND EXISTS ${MLX_LIB_DIR}/mlx.metallib)
  file(COPY ${MLX_LIB_DIR}/mlx.metallib DESTINATION ${CMAKE_BINARY_DIR})
endif()

# ---- ONNX Runtime (C++; pipeline backend) -----------------------------------
# Preferred: a CPU-only *static* libonnxruntime.a built from source
# (scripts/build_onnxruntime_static.sh) so the binary carries no libonnxruntime.dylib at
# runtime. MUST be linked with -force_load so ORT's static kernel-registration initializers
# survive (otherwise inference fails with "no suitable kernel"). Fallback: the pip dylib.
set(ORT_STATIC_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-static)
set(ORT_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
set(ORT_STATIC FALSE)
set(_ort_real "")
if(EXISTS ${ORT_STATIC_DIR}/lib/libonnxruntime.a)
  add_library(onnxruntime INTERFACE)
  target_include_directories(onnxruntime INTERFACE ${ORT_STATIC_DIR}/include)
  # force_load ONLY the providers archive (the CPU-kernel static-registration initializers);
  # link the rest normally so the linker de-dupes onnx/onnx_proto/protobuf overlaps.
  target_link_libraries(onnxruntime INTERFACE
    "-Wl,-force_load,${ORT_STATIC_DIR}/lib/libonnxruntime_providers.a"
    ${ORT_STATIC_DIR}/lib/libonnxruntime.a
    "-framework Foundation")
  set(HAVE_ORT TRUE)
  set(ORT_STATIC TRUE)
  message(STATUS "Using static ONNX Runtime at ${ORT_STATIC_DIR}")
elseif(EXISTS ${ORT_DIR}/lib/libonnxruntime.dylib)
  add_library(onnxruntime SHARED IMPORTED)
  file(GLOB _ort_dylib ${ORT_DIR}/lib/libonnxruntime.*.dylib)
  list(GET _ort_dylib 0 _ort_real)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${_ort_real}
    INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/include)
  set(HAVE_ORT TRUE)
  message(STATUS "Found ONNX Runtime at ${ORT_DIR}")
else()
  message(WARNING "ONNX Runtime not found; run scripts/build_onnxruntime_static.sh (static) "
                  "or scripts/fetch_onnxruntime.sh (dylib). Pipeline targets disabled.")
  set(HAVE_ORT FALSE)
endif()

# ---- MNN (static; pipeline acceleration) ------------------------------------
# Built static via scripts/build_mnn_static.sh (MNN 3.6.0, CPU + Metal + CoreML). 4 pipeline
# models run faster on MNN with golden-verified parity (ocr_det, ocr_rec, table_cls, unet); for
# these mumodel ships only the `.mnn` (the `.onnx` is archived in orgmodel), so the binary
# REQUIRES MNN to run them. The other models (layout, slanet, mfr_*) stay on ORT. Each wrapper
# uses a `<model>.mnn` if present, else the `.onnx` via ORT.
#
# The backend is chosen at runtime via MINERU_MNN_BACKEND (cpu|metal|coreml|auto), default cpu
# (see src/pipeline/mnn_runner.cpp). Metal gives large GPU speedups on Apple Silicon (unet ~18x,
# ocr_rec ~8x, ocr_det ~2.4x; CoreML currently falls back to CPU). The Metal & CoreML backends
# are compiled into libMNN.a, so the final binary must link the Apple GPU/ANE frameworks below;
# Metal shaders are embedded as source (compiled at runtime), so no external mnn.metallib is
# needed and the binary stays relocatable/dylib-free.
set(MNN_STATIC_DIR ${CMAKE_SOURCE_DIR}/third_party/mnn-static)
if(EXISTS ${MNN_STATIC_DIR}/lib/libMNN.a)
  add_library(mnn INTERFACE)
  target_include_directories(mnn INTERFACE ${MNN_STATIC_DIR}/include)
  target_link_libraries(mnn INTERFACE
    ${MNN_STATIC_DIR}/lib/libMNN.a
    "-framework Foundation" "-framework CoreFoundation" "-framework Accelerate"
    "-framework Metal" "-framework CoreGraphics"          # MNN Metal backend (GPU)
    "-framework CoreML" "-framework CoreVideo")            # MNN CoreML backend (ANE/GPU)
  set(HAVE_MNN TRUE)
  message(STATUS "Using static MNN at ${MNN_STATIC_DIR} (CPU/Metal/CoreML; pipeline accel)")
else()
  set(HAVE_MNN FALSE)
endif()

# ---- OpenCV (static; bit-exact cv2 ops for the wired-table UNet path) ---------
# Built static from source (4.13.0) via scripts/build_opencv_static.sh into
# third_party/opencv/ so the binary links core+imgproc as static archives and carries
# NO opencv dylib at runtime (no Homebrew dependency). If the static build is absent,
# OpenCV is simply disabled and the wired-table path falls back to reimplemented cv ops
# — we deliberately do NOT fall back to a Homebrew dylib, to keep the binary self-contained.
set(OPENCV_STATIC_DIR ${CMAKE_SOURCE_DIR}/third_party/opencv)
if(EXISTS ${OPENCV_STATIC_DIR}/lib/libopencv_core.a)
  set(OpenCV_DIR ${OPENCV_STATIC_DIR}/lib/cmake/opencv4)
  find_package(OpenCV QUIET COMPONENTS core imgproc NO_DEFAULT_PATH PATHS ${OpenCV_DIR})
endif()
if(OpenCV_FOUND)
  set(HAVE_OPENCV TRUE)
  message(STATUS "Found static OpenCV ${OpenCV_VERSION} at ${OPENCV_STATIC_DIR}")
else()
  set(HAVE_OPENCV FALSE)
  message(STATUS "static OpenCV not built (scripts/build_opencv_static.sh); "
                 "wired-table UNet path uses reimplemented cv ops")
endif()

# ---- Core library ------------------------------------------------------------
add_library(mineru_core STATIC
  src/core/middle_json.cpp
  src/core/pipeline_assemble.cpp
  src/core/table_merge.cpp
  src/output/text_utils.cpp
  src/output/mkcontent.cpp
  src/io/image_write.cpp
  src/io/image_to_pdf.cpp
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

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/table_merge_golden.json)
  add_executable(test_table_merge tests/test_table_merge.cpp)
  target_include_directories(test_table_merge PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_table_merge PRIVATE mineru_core)
  add_test(NAME table_merge COMMAND test_table_merge ${CMAKE_SOURCE_DIR}/tests/golden)
endif()

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
  add_library(mineru_pdf STATIC src/io/pdf.cpp src/io/draw_bbox.cpp)
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
  set(MINERU_PIPELINE_SRC src/pipeline/layout_det.cpp src/pipeline/table_cls.cpp src/pipeline/ocr_rec.cpp src/pipeline/ocr_det.cpp src/pipeline/ocr.cpp src/pipeline/formula_rec.cpp src/pipeline/table_rec.cpp src/pipeline/post_ocr.cpp src/pipeline/pipeline_driver.cpp src/pipeline/wired_table.cpp src/pipeline/cv_resize.cpp src/pipeline/mnn_runner.cpp)
  if(HAVE_MLX)
    list(APPEND MINERU_PIPELINE_SRC src/pipeline/mfr_mlx_decoder.cpp)  # MLX/Metal MFR decoder
  endif()
  add_library(mineru_pipeline STATIC ${MINERU_PIPELINE_SRC})
  target_include_directories(mineru_pipeline PUBLIC ${CMAKE_SOURCE_DIR}/include)
  if(HAVE_MLX)
    target_link_libraries(mineru_pipeline PUBLIC mlx)
    target_compile_definitions(mineru_pipeline PUBLIC MINERU_HAVE_MLX=1)
  endif()
  # MNN must come BEFORE ONNX Runtime on the link line: both statically bundle flatbuffers, and
  # ORT is -force_load'd (pulls in its flatbuffers symbols first). If ORT wins, MNN parses .mnn
  # with ORT's flatbuffers and Module::load fails ("Invalidate buffer"). Linking MNN first makes
  # MNN's flatbuffers symbols resolve for libMNN.
  if(HAVE_MNN)
    target_link_libraries(mineru_pipeline PUBLIC mnn)
    target_compile_definitions(mineru_pipeline PUBLIC MINERU_HAVE_MNN=1)
  endif()
  target_link_libraries(mineru_pipeline PUBLIC onnxruntime mineru_vlm mineru_core thirdparty_headers)
  if(HAVE_OPENCV)
    target_link_libraries(mineru_pipeline PUBLIC ${OpenCV_LIBS})
    target_include_directories(mineru_pipeline PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(mineru_pipeline PUBLIC MINERU_HAVE_OPENCV=1)
  endif()
endif()

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(test_tokenizer PRIVATE mineru_vlm)
add_test(NAME tokenizer
  COMMAND test_tokenizer ${CMAKE_SOURCE_DIR}/mumodel/MinerU2.5-tokenizer
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

  # ---- MFR MLX decoder validation (vs ONNX reference) -----------------------
  if(EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_decoder.safetensors)
    add_executable(test_mfr_mlx tests/test_mfr_mlx.cpp src/pipeline/mfr_mlx_decoder.cpp)
    target_include_directories(test_mfr_mlx PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(test_mfr_mlx PRIVATE mlx thirdparty_headers)
    set_target_properties(test_mfr_mlx PROPERTIES BUILD_RPATH ${MLX_LIB_DIR})
    add_test(NAME mfr_mlx COMMAND test_mfr_mlx
             ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_decoder.safetensors
             ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_decoder_config.json)
  endif()

  # ---- Qwen2-VL model (MLX) -------------------------------------------------
  add_library(mineru_qwen STATIC src/vlm/qwen2_llm.cpp)
  target_include_directories(mineru_qwen PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(mineru_qwen PUBLIC mlx thirdparty_headers)

  # ---- mlx-mineru CLI -------------------------------------------------------
  if(HAVE_PDFIUM)
    # Embed the built web UI (web/dist) into a generated source (stub if absent).
    execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/embed_web.py)
    add_executable(mlx-mineru src/cli/main.cpp src/cli/web_assets.gen.cpp)
    target_link_libraries(mlx-mineru PRIVATE mineru_qwen mineru_vlm mineru_pdf mineru_core mineru_getmodel)
    # @loader_path first so a relocated bundle (binary + colocated dylibs) is
    # self-contained; the absolute third_party dirs follow for in-tree dev runs.
    set(MLX_MINERU_RPATH "@loader_path;${MLX_LIB_DIR};${PDFIUM_DIR}/lib")
    if(HAVE_ORT)
      # Add the native pipeline backend (--backend pipeline) to the CLI.
      target_link_libraries(mlx-mineru PRIVATE mineru_pipeline)
      target_compile_definitions(mlx-mineru PRIVATE MINERU_HAVE_PIPELINE=1)
      set(MLX_MINERU_RPATH "${MLX_MINERU_RPATH};${ORT_DIR}/lib")
    endif()
    set_target_properties(mlx-mineru PROPERTIES BUILD_RPATH "${MLX_MINERU_RPATH}")

    # Bundle the remaining runtime dylibs (those with no static build) next to the executable
    # so the binary runs with no external/build-tree dependency — copy + the @loader_path rpath
    # above make `build/mlx-mineru` + its colocated dylibs relocatable. pdfium is linked
    # statically when available, so it is only copied in the dylib-fallback case.
    if(NOT PDFIUM_STATIC)
      add_custom_command(TARGET mlx-mineru POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PDFIUM_DIR}/lib/libpdfium.dylib $<TARGET_FILE_DIR:mlx-mineru>
        VERBATIM)
    endif()
    if(HAVE_ORT AND NOT ORT_STATIC)
      add_custom_command(TARGET mlx-mineru POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_ort_real} $<TARGET_FILE_DIR:mlx-mineru>
        VERBATIM)
    endif()
  endif()

  set(MODEL_ST ${CMAKE_SOURCE_DIR}/mumodel/MinerU2.5-tokenizer/model.safetensors)
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
if(HAVE_ORT AND EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout/layout.onnx)
  add_executable(test_onnx_smoke tests/test_onnx_smoke.cpp)
  target_include_directories(test_onnx_smoke PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(test_onnx_smoke PRIVATE onnxruntime)
  set_target_properties(test_onnx_smoke PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
  add_test(NAME onnx_smoke
    COMMAND test_onnx_smoke ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout/layout.onnx)
endif()

# ---- Pipeline backend tests (library defined earlier) -----------------------
if(HAVE_ORT)
  if(EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout/layout.onnx OR EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout/layout.mnn)
    add_executable(test_layout_det tests/test_layout_det.cpp)
    target_include_directories(test_layout_det PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_layout_det PRIVATE mineru_pipeline)
    set_target_properties(test_layout_det PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME layout_det
      COMMAND test_layout_det ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout
              ${CMAKE_SOURCE_DIR}/tests/golden)

    if(HAVE_PDFIUM AND EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/pipeline_p0_model.json)
      add_executable(test_layout_order tests/test_layout_order.cpp)
      target_include_directories(test_layout_order PRIVATE ${CMAKE_SOURCE_DIR}/tests)
      target_link_libraries(test_layout_order PRIVATE mineru_pipeline mineru_pdf)
      set_target_properties(test_layout_order PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
      add_test(NAME layout_order COMMAND test_layout_order ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout
                ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/a.pdf)
    endif()
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx OR EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabCls/PP-LCNet_x1_0_table_cls.mnn)
    add_executable(test_tab_cls tests/test_tab_cls.cpp)
    target_include_directories(test_tab_cls PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_tab_cls PRIVATE mineru_pipeline)
    set_target_properties(test_tab_cls PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME tab_cls
      COMMAND test_tab_cls ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx
              ${CMAKE_SOURCE_DIR}/tests/golden)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx OR EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.mnn)
    add_executable(test_ocr_rec tests/test_ocr_rec.cpp)
    target_include_directories(test_ocr_rec PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_ocr_rec PRIVATE mineru_pipeline)
    set_target_properties(test_ocr_rec PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME ocr_rec
      COMMAND test_ocr_rec ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/tests/golden)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/ocr_det.json)
    add_executable(test_ocr_det tests/test_ocr_det.cpp)
    target_include_directories(test_ocr_det PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_ocr_det PRIVATE mineru_pipeline)
    set_target_properties(test_ocr_det PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME ocr_det COMMAND test_ocr_det ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_det.onnx)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/ocr_page.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_det.onnx AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx)
    add_executable(test_ocr_page tests/test_ocr_page.cpp)
    target_include_directories(test_ocr_page PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_ocr_page PRIVATE mineru_pipeline)
    set_target_properties(test_ocr_page PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME ocr_page COMMAND test_ocr_page ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_det.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ppocrv6_dict.txt)
  endif()

  if(HAVE_PDFIUM AND EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/pipeline_p0_model.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx)
    add_executable(test_pipeline_textfill tests/test_pipeline_textfill.cpp)
    target_include_directories(test_pipeline_textfill PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_textfill PRIVATE mineru_pipeline mineru_core mineru_pdf)
    set_target_properties(test_pipeline_textfill PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
    add_test(NAME pipeline_textfill COMMAND test_pipeline_textfill ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/a.pdf)

    add_executable(test_pipeline_md tests/test_pipeline_md.cpp)
    target_include_directories(test_pipeline_md PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_md PRIVATE mineru_pipeline mineru_core mineru_pdf)
    set_target_properties(test_pipeline_md PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
    add_test(NAME pipeline_md COMMAND test_pipeline_md ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/a.pdf)

    add_executable(test_pipeline_multipage tests/test_pipeline_multipage.cpp)
    target_include_directories(test_pipeline_multipage PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_pipeline_multipage PRIVATE mineru_pipeline mineru_core mineru_pdf)
    set_target_properties(test_pipeline_multipage PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
    add_test(NAME pipeline_multipage COMMAND test_pipeline_multipage ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_rec.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ppocrv6_dict.txt
              ${CMAKE_SOURCE_DIR}/a.pdf)

    if(EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/Layout/layout.onnx AND
       EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/OCR/ocr_det.onnx)
      add_executable(test_pipeline_scratch tests/test_pipeline_scratch.cpp)
      target_include_directories(test_pipeline_scratch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
      target_link_libraries(test_pipeline_scratch PRIVATE mineru_pipeline mineru_core mineru_pdf)
      set_target_properties(test_pipeline_scratch PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
      add_test(NAME pipeline_scratch COMMAND test_pipeline_scratch ${CMAKE_SOURCE_DIR}/tests/golden
                ${CMAKE_SOURCE_DIR}/mumodel/pipeline ${CMAKE_SOURCE_DIR}/a.pdf)

      if(EXISTS ${CMAKE_SOURCE_DIR}/tests/demo1.pdf AND
         EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/demo1_p2_model.json AND
         EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_encoder.onnx)
        add_executable(test_pipeline_formula_scratch tests/test_pipeline_formula_scratch.cpp)
        target_include_directories(test_pipeline_formula_scratch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        target_link_libraries(test_pipeline_formula_scratch PRIVATE mineru_pipeline mineru_core mineru_pdf)
        set_target_properties(test_pipeline_formula_scratch PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
        add_test(NAME pipeline_formula_scratch COMMAND test_pipeline_formula_scratch
                  ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/mumodel/pipeline
                  ${CMAKE_SOURCE_DIR}/tests/demo1.pdf)
      endif()

      if(EXISTS ${CMAKE_SOURCE_DIR}/tests/demo1.pdf AND
         EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/demo1_p5_model.json AND
         EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabRec/SlanetPlus/slanet-plus.onnx)
        add_executable(test_pipeline_table_scratch tests/test_pipeline_table_scratch.cpp)
        target_include_directories(test_pipeline_table_scratch PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        target_link_libraries(test_pipeline_table_scratch PRIVATE mineru_pipeline mineru_core mineru_pdf)
        set_target_properties(test_pipeline_table_scratch PROPERTIES BUILD_RPATH "${ORT_DIR}/lib;${PDFIUM_DIR}/lib")
        add_test(NAME pipeline_table_scratch COMMAND test_pipeline_table_scratch
                  ${CMAKE_SOURCE_DIR}/tests/golden ${CMAKE_SOURCE_DIR}/mumodel/pipeline
                  ${CMAKE_SOURCE_DIR}/tests/demo1.pdf)
      endif()
    endif()
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/mfr.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_encoder.onnx AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_decoder.onnx)
    add_executable(test_mfr tests/test_mfr.cpp)
    target_include_directories(test_mfr PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_mfr PRIVATE mineru_pipeline)
    set_target_properties(test_mfr PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME mfr COMMAND test_mfr ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_encoder.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_decoder.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/MFR/mfr_vocab.txt)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/wired_table.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabRec/UnetStructure/unet.onnx)
    add_executable(test_wired_table tests/test_wired_table.cpp)
    target_include_directories(test_wired_table PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_wired_table PRIVATE mineru_pipeline)
    set_target_properties(test_wired_table PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME wired_table COMMAND test_wired_table ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabRec/UnetStructure/unet.onnx)
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/tests/golden/table_struct.json AND
     EXISTS ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabRec/SlanetPlus/slanet-plus.onnx)
    add_executable(test_table tests/test_table.cpp)
    target_include_directories(test_table PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_table PRIVATE mineru_pipeline)
    set_target_properties(test_table PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
    add_test(NAME table COMMAND test_table ${CMAKE_SOURCE_DIR}/tests/golden
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabRec/SlanetPlus/slanet-plus.onnx
              ${CMAKE_SOURCE_DIR}/mumodel/pipeline/TabRec/SlanetPlus/table_structure_dict.txt)
  endif()
endif()

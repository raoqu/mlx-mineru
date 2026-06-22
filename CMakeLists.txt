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

# ---- Core library ------------------------------------------------------------
add_library(mineru_core STATIC
  src/core/middle_json.cpp
)
target_include_directories(mineru_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
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

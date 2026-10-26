cmake_minimum_required(VERSION 3.20)
project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Core library: everything except the CLI entry point, so tests can link it.
add_library(tinyinfer_core
  src/model.cpp
  src/ops.cpp
)
target_include_directories(tinyinfer_core PUBLIC include)

add_executable(tinyinfer src/main.cpp)
target_link_libraries(tinyinfer PRIVATE tinyinfer_core)

# Tests
option(TINYINFER_BUILD_TESTS "Build unit tests" ON)
if(TINYINFER_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(tinyinfer_tests
    tests/test_loader.cpp
    tests/test_forward.cpp
  )
  target_link_libraries(tinyinfer_tests PRIVATE tinyinfer_core GTest::gtest_main)
  # Tests look for exported weights relative to the repo root.
  target_compile_definitions(tinyinfer_tests PRIVATE
    TINYINFER_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")

  include(GoogleTest)
  gtest_discover_tests(tinyinfer_tests)
endif()

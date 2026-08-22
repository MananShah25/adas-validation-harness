cmake_minimum_required(VERSION 3.18)
project(adas_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ADAS_BUILD_TESTS "Build the GoogleTest suite" ON)
option(ADAS_BUILD_PYTHON_MODULE "Build the pybind11 extension module" ON)

# ---------------------------------------------------------------------
# Core library: pure C++, no Python dependency at all.
# Keeping the algorithms free of any binding code means the GoogleTest
# suite exercises exactly the same objects the Python module calls, and
# the core could be reused in a non-Python (e.g. embedded) context.
# ---------------------------------------------------------------------
add_library(adas_core_lib STATIC
  src/metrics.cpp
  src/controllers.cpp
)

target_include_directories(adas_core_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(adas_core_lib PRIVATE -Wall -Wextra -Wpedantic)

# Disable floating-point contraction (fused multiply-add).
#
# This is REQUIRED for bit-exact parity with the Python reference, and it was
# found by the parity test rather than by inspection. On ARM64, clang defaults
# to contracting expressions like `-0.3 * offset - 0.5 * heading` into a single
# FMA instruction, which rounds ONCE instead of twice. That is genuinely more
# accurate, but it is not what CPython does -- CPython evaluates each multiply
# and the subtraction as separate IEEE-754 double operations, rounding at each
# step. The two answers differ in the last ulp:
#
#     FMA contracted : -0.13999999999999999
#     separate ops   : -0.14000000000000001   <- CPython, and now C++ too
#
# For a validation harness this matters: the harness compares measured values
# against fixed thresholds, so a last-ulp difference can flip a borderline
# PASS/FAIL. Determinism against the reference beats marginally better
# accuracy here, so contraction is switched off deliberately.
target_compile_options(adas_core_lib PRIVATE -ffp-contract=off)

# Position-independent code is required because this static library gets
# linked into a shared object (the Python extension module).
set_target_properties(adas_core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------
# Python extension module (pybind11)
# ---------------------------------------------------------------------
if(ADAS_BUILD_PYTHON_MODULE)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(adas_core src/bindings.cpp)
  target_link_libraries(adas_core PRIVATE adas_core_lib)
  target_compile_options(adas_core PRIVATE -Wall -Wextra -Wpedantic)

  # Drop the built .so next to the Python packages so `import adas_core`
  # works from the repo root without installing or setting PYTHONPATH.
  set_target_properties(adas_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
  )
endif()

# ---------------------------------------------------------------------
# GoogleTest suite
# ---------------------------------------------------------------------
if(ADAS_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(adas_core_tests
    tests/test_metrics.cpp
    tests/test_controllers.cpp
  )
  target_link_libraries(adas_core_tests PRIVATE adas_core_lib GTest::gtest_main)
  target_compile_options(adas_core_tests PRIVATE -Wall -Wextra -Wpedantic)

  include(GoogleTest)
  gtest_discover_tests(adas_core_tests)
endif()

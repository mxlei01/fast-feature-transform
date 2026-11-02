cmake_minimum_required(VERSION 3.28)

# One Arrow strategy everywhere: a pinned vcpkg (make vcpkg) builds a private
# static Arrow from vcpkg.json. The checkout and the install tree live outside
# the repository, in FFT_DEPS_DIR (default ~/.cache/fast-feature-transform,
# also read from the environment), so a checkout in a synced folder stays
# small. The vcpkg toolchain is set here, before project() as CMake requires,
# so a plain `uv sync` needs no CMake arguments.
if(NOT DEFINED FFT_DEPS_DIR)
  set(FFT_DEPS_DIR "$ENV{FFT_DEPS_DIR}")
endif()
if(FFT_DEPS_DIR STREQUAL "")
  set(FFT_DEPS_DIR "$ENV{HOME}/.cache/fast-feature-transform")
endif()
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(EXISTS "${FFT_DEPS_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${FFT_DEPS_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  else()
    message(FATAL_ERROR "No vcpkg checkout at ${FFT_DEPS_DIR}/vcpkg: run `make vcpkg` first (or set FFT_DEPS_DIR)")
  endif()
endif()
# One install tree for every build directory (Python module, C++ tests, CI):
# `make vcpkg` populates it once and each configure only verifies it.
set(VCPKG_INSTALLED_DIR "${FFT_DEPS_DIR}/vcpkg_installed")

project(fast_feature_transform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ON builds the native Catch2 test binary (no Python anywhere in the build);
# OFF (the default, what scikit-build-core drives) builds the Python module.
option(FFT_BUILD_TESTS "Build the C++ unit tests instead of the Python module" OFF)
option(FFT_ENABLE_SANITIZERS "Instrument native C++ tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FFT_ENABLE_THREAD_SANITIZER "Instrument native C++ tests with ThreadSanitizer" OFF)

if(FFT_ENABLE_SANITIZERS OR FFT_ENABLE_THREAD_SANITIZER)
  if(NOT FFT_BUILD_TESTS)
    message(FATAL_ERROR "The sanitizer options require FFT_BUILD_TESTS=ON; use a separate native test build")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    message(FATAL_ERROR "The sanitizer options require GCC or Clang")
  endif()
  if(FFT_ENABLE_SANITIZERS AND FFT_ENABLE_THREAD_SANITIZER)
    message(FATAL_ERROR "ThreadSanitizer cannot share a build with AddressSanitizer; use separate build directories")
  endif()
endif()

# Debug builds — the native test and sanitizer builds — run with the standard
# library's own precondition checks, so an index past size() or front() on an
# empty container fails a test instead of passing through as undefined
# behavior. libc++ also checks that a comparator handed to std::sort is a
# strict weak ordering; libstdc++'s assertions do not check comparators.
# Each macro is ignored by the other library and neither changes the ABI, so
# the prebuilt static Arrow still links. Release builds, the Python module's
# and the native benchmarks', are unchanged.
add_compile_definitions(
  $<$<CONFIG:Debug>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG>
  $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)

# Apply these settings only to our core and binding translation units. Linking
# an INTERFACE target privately does not change how third-party libraries build.
add_library(fft_project_options INTERFACE)
target_compile_options(fft_project_options INTERFACE
  -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
  -Wdouble-promotion -Wnon-virtual-dtor -Wnull-dereference
  # Preserve separate roundings for multiply then add across compilers and
  # match the reference implementations' arithmetic.
  -ffp-contract=off)

function(fft_enable_sanitizers target)
  if(FFT_ENABLE_SANITIZERS)
    target_compile_options(${target} PRIVATE
      -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  elseif(FFT_ENABLE_THREAD_SANITIZER)
    target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=thread)
  endif()
endfunction()

if(NOT FFT_BUILD_TESTS)
  find_package(Python 3.14 COMPONENTS Interpreter Development.Module REQUIRED)

  # nanobind is installed in the build environment; ask it where its CMake config lives.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(nanobind CONFIG REQUIRED)
endif()

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(nlohmann_json)

# Flat (robin-hood-family) hash map for the kernels' category and group tables.
FetchContent_Declare(unordered_dense
  URL https://github.com/martinus/unordered_dense/archive/refs/tags/v4.5.0.tar.gz
  URL_HASH SHA256=2364ce4bc4c23bd02549bbb3a7572d881684cd46057f3737fd53be53669743aa
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(unordered_dense)

# The package version (git-tag-derived via setuptools_scm; scikit-build-core
# hands it to CMake) is baked into the binary so every artifact manifest
# records which build fitted it. The FULL form keeps the .devN+g<sha> suffix
# of untagged builds; 0.0.0 means built outside the package (C++ tests).
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(FFT_LIBRARY_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
elseif(DEFINED SKBUILD_PROJECT_VERSION)
  set(FFT_LIBRARY_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(FFT_LIBRARY_VERSION "0.0.0")
endif()

add_library(fftcore STATIC
  src/fastfeaturetransform/spill.cpp
  src/fastfeaturetransform/io.cpp
  src/fastfeaturetransform/kernel.cpp
  src/fastfeaturetransform/kernels/basic.cpp
  src/fastfeaturetransform/kernels/count.cpp
  src/fastfeaturetransform/kernels/ordinal.cpp
  src/fastfeaturetransform/kernels/basen.cpp
  src/fastfeaturetransform/kernels/contrast.cpp
  src/fastfeaturetransform/kernels/group_agg.cpp
  src/fastfeaturetransform/kernels/hash.cpp
  src/fastfeaturetransform/kernels/target.cpp
  src/fastfeaturetransform/kernels/shrinkage.cpp
  src/fastfeaturetransform/kernels/quantile.cpp
  src/fastfeaturetransform/kernels/scale.cpp
  src/fastfeaturetransform/kernels/binning.cpp
  src/fastfeaturetransform/kernels/expand.cpp
  src/fastfeaturetransform/engine.cpp)
target_compile_definitions(fftcore PRIVATE FFT_LIBRARY_VERSION="${FFT_LIBRARY_VERSION}")
target_include_directories(fftcore PUBLIC src)
target_link_libraries(fftcore PUBLIC Arrow::arrow_static Parquet::parquet_static nlohmann_json::nlohmann_json
                      unordered_dense::unordered_dense)
target_link_libraries(fftcore PRIVATE fft_project_options)
fft_enable_sanitizers(fftcore)

# Built only when explicitly requested; measures the same production kernels
# without Python, fitting, or Parquet I/O. Use a Release configuration.
add_executable(fft_scaler_benchmark EXCLUDE_FROM_ALL benchmark/scalers.cpp)
target_compile_options(fft_scaler_benchmark PRIVATE -ffp-contract=off)
target_link_libraries(fft_scaler_benchmark PRIVATE fftcore)
fft_enable_sanitizers(fft_scaler_benchmark)

# Also built only on request; use a Release configuration. It times every
# kernel's fit and transform without Python or Parquet I/O. Its sampler uses
# Mach thread calls and reads arm64 registers, so it compiles only on arm64
# macOS, although the target is declared on any Apple platform
# (README.md, Profiling).
if(APPLE)
  add_executable(fft_kernel_benchmark EXCLUDE_FROM_ALL benchmark/kernels.cpp)
  target_compile_options(fft_kernel_benchmark PRIVATE -ffp-contract=off)
  target_link_libraries(fft_kernel_benchmark PRIVATE fftcore)
  fft_enable_sanitizers(fft_kernel_benchmark)
endif()

if(FFT_BUILD_TESTS)
  FetchContent_Declare(catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(catch2)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  enable_testing()

  add_executable(fft_tests
    src/fastfeaturetransform/spill_test.cpp
    src/fastfeaturetransform/kernel_test.cpp
    src/fastfeaturetransform/kernels/count_test.cpp
    src/fastfeaturetransform/kernels/ordinal_test.cpp
    src/fastfeaturetransform/kernels/output_test.cpp
    src/fastfeaturetransform/kernels/basen_test.cpp
    src/fastfeaturetransform/kernels/contrast_test.cpp
    src/fastfeaturetransform/kernels/hash_test.cpp
    src/fastfeaturetransform/kernels/target_test.cpp
    src/fastfeaturetransform/kernels/shrinkage_test.cpp
    src/fastfeaturetransform/kernels/quantile_test.cpp
    src/fastfeaturetransform/kernels/scale_test.cpp
    src/fastfeaturetransform/kernels/scale_simd_test.cpp
    src/fastfeaturetransform/kernels/binning_test.cpp
    src/fastfeaturetransform/kernels/expand_test.cpp
    src/fastfeaturetransform/engine_test.cpp
    src/fastfeaturetransform/io_test.cpp)
  # The tests compute expected values of their own, so they need the same
  # contraction rule the library ships with or an assertion could pass here and
  # fail in the shipped extension.
  target_compile_options(fft_tests PRIVATE -ffp-contract=off)
  target_link_libraries(fft_tests PRIVATE fftcore Catch2::Catch2WithMain)
  fft_enable_sanitizers(fft_tests)
  include(Catch)
  catch_discover_tests(fft_tests)
  if(NOT FFT_ENABLE_SANITIZERS AND NOT FFT_ENABLE_THREAD_SANITIZER)
    # This isolated failure test replaces global allocation functions; keep it
    # separate from both Catch2 and the sanitizer runtimes' allocator hooks.
    add_executable(fft_spill_cleanup_test src/fastfeaturetransform/spill_cleanup_test.cpp)
    target_compile_options(fft_spill_cleanup_test PRIVATE -ffp-contract=off)
    target_link_libraries(fft_spill_cleanup_test PRIVATE fftcore)
    add_test(NAME spill_cleanup_allocation_failure COMMAND fft_spill_cleanup_test)
  endif()
else()
  # NOMINSIZE: nanobind otherwise compiles the module at -Os, which leaves its
  # reference-counting and casting helpers out of line. The record path is
  # nothing but those helpers around a kernel call, and it is the serving path.
  nanobind_add_module(_core FREE_THREADED NOMINSIZE src/bindings/module.cpp)
  target_link_libraries(_core PRIVATE fftcore fft_project_options)
  # Export only the module init symbol: our static Arrow stays private, so its
  # (weak) symbols can never coalesce with pyarrow's Arrow in the same process.
  if(APPLE)
    target_link_options(_core PRIVATE "LINKER:-exported_symbol,_PyInit__core")
  else()
    target_link_options(_core PRIVATE "LINKER:--exclude-libs,ALL")
  endif()

  install(TARGETS _core LIBRARY DESTINATION fast_feature_transform)

  nanobind_add_stub(_core_stub
    MODULE _core
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
    PYTHON_PATH "$<TARGET_FILE_DIR:_core>"
    MARKER_FILE "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
    DEPENDS _core)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi" "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
          DESTINATION fast_feature_transform)
endif()

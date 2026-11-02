# vcpkg's own arm64-osx triplet, plus the two settings below.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES arm64)
# The wheels support macOS 26 and later, the first whose C++ library parses doubles
# with std::from_chars.
set(VCPKG_OSX_DEPLOYMENT_TARGET 26.0)

# Parquet's modular encryption is all that links OpenSSL into the extension, and this
# package reads no encrypted files. vcpkg's arrow port turns it on with the parquet
# feature, so turn it off here.
if(PORT STREQUAL "arrow")
  list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DPARQUET_REQUIRE_ENCRYPTION=OFF)
endif()

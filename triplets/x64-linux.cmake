# vcpkg's own x64-linux triplet, plus the setting below.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Parquet's modular encryption is all that links OpenSSL into the extension, and this
# package reads no encrypted files. vcpkg's arrow port turns it on with the parquet
# feature, so turn it off here.
if(PORT STREQUAL "arrow")
  list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DPARQUET_REQUIRE_ENCRYPTION=OFF)
endif()

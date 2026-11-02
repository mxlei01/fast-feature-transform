# vcpkg's own x64-linux triplet, plus the settings below.
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

# uriparser 1.0.2 decides it can call reallocarray because the function links,
# then compiles as strict C99, where glibc older than 2.29 (the manylinux_2_28
# wheel images have 2.28) does not declare it, and GCC 14 rejects the call.
# Turning the check off makes it use its own overflow-checked realloc;
# uriparser 1.0.3 fixes the check.
if(PORT STREQUAL "uriparser")
  list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DHAVE_REALLOCARRAY=OFF)
endif()

# Thrift 0.25.0 passes --file-prefix-map to bison, an option bison 3.7 added,
# and the manylinux_2_28 images have bison 3.0.4. Only Thrift's compiler uses
# bison, and Arrow needs just the library, so the compiler is not built.
if(PORT STREQUAL "thrift")
  list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DBUILD_COMPILER=OFF)
endif()

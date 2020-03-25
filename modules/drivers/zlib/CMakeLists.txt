#[[ If we wanted to just find a pre-installed version, we could use something like:
set(ZLIB_ROOT "${PROJECT_SOURCE_DIR}/modules/drivers/zlib")
include(FindZLIB)
find_package(ZLIB REQUIRED)
# This would create the ZLIB::ZLIB target.  Then use the following in the dependent projects:
if (ZLIB_FOUND)
    target_include_directories("${TARGET_NAME}" ${ZLIB_INCLUDE_DIRS})
    target_link_libraries("${TARGET_NAME}" ${ZLIB_LIBRARIES})
endif()
#]]

set(TARGET_NAME ZLIB)

# Build with ZLIB's build system
coda_add_driver(
    ${TARGET_NAME}
    "zlib-1.2.7.tar"
    "SHA256=df0ba700245709ea54a18536ed266c2f0a0b3e4c3eb029abc42bf48a087901c5")

if (WIN32)
    set(ZLIB_LIBRARY zlib CACHE STRING "name of the zlib library")
else()
    set(ZLIB_LIBRARY z CACHE STRING "name of the zlib library")
endif()

include("minizip.cmake")

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

set(TARGET_NAME "ZLIB")

# Build with ZLIB's build system
coda_add_driver(
    "${TARGET_NAME}"
    "zlib-1.2.7.tar"
    "SHA256=df0ba700245709ea54a18536ed266c2f0a0b3e4c3eb029abc42bf48a087901c5"
    TRUE)

include("coda-oss_minizip.cmake")

target_include_directories(zlib PUBLIC
    $<BUILD_INTERFACE:${coda-oss_zlib_SOURCE_DIR}
    $<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>)
target_include_directories(zlibstatic PUBLIC
    $<BUILD_INTERFACE:${coda-oss_zlib_SOURCE_DIR}
    $<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>)

# disable some unneeded targets from zlib
foreach(tgt example minigzip example64 minigzip64)
    if(TARGET ${tgt})
        set_target_properties(${tgt} PROPERTIES
            EXCLUDE_FROM_ALL 1
            EXCLUDE_FROM_DEFAULT_BUILD 1)
    endif()
endforeach()

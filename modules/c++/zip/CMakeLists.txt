set("TARGET_NAME"		"zip")
set("MAINTAINER"		"jmrandol@users.sourceforge.net")
set("TARGET_VERSION"	"1.0")
set("MODULE_DEPS"		"io" "ZLIB::ZLIB")
#xxx set("USELIB_CHECK"		"MINIZIP" "ZIP")

#[[
set(ZLIB_ROOT "${PROJECT_SOURCE_DIR}/modules/drivers/zlib")
message(" ${PROJECT_SOURCE_DIR}/modules/drivers/zlib/zlib-1.2.7")
include(FindZLIB)
find_package(ZLIB REQUIRED)
]]

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (ZLIB_FOUND)
    target_include_directories("${TARGET_NAME}" ${ZLIB_INCLUDE_DIRS})
    target_link_libraries("${TARGET_NAME}" ${ZLIB_LIBRARIES})
endif()
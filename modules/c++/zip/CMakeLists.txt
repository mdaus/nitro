set(TARGET_NAME "zip")
set(MODULE_DEPS "io" "zlib" "coda-oss_minizip")
#xxx If we want to use FindZLib instead:
#set("MODULE_DEPS" "io" "ZLIB::ZLIB" "coda-oss_minizip")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (ZLIB_FOUND)
    target_include_directories("${TARGET_NAME}" ${ZLIB_INCLUDE_DIRS})
    target_link_libraries("${TARGET_NAME}" ${ZLIB_LIBRARIES})
endif()

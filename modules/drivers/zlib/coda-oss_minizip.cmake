# Some of our code depends on code from a 'minizip' project that is distributed
#   with zlib.  However, we don't want the whole project; just a couple of
#   files, bundled into a library.
set(OUTPUT_NAME "minizip")
set(TARGET_NAME "${CMAKE_PROJECT_NAME}_${OUTPUT_NAME}")
set(MODULE_DEPS "z")
set(EXTRA_DEPS "coda-oss_ZLIB")

# The 3P's own build process makes an executable, which we can't re-use.
# Set up a small custom project to build a library containing the subset
# of this 3P's files that we want to reuse.

#xxx Find the zlib dir directly?
set(source_dir "${coda-oss_zlib_SOURCE_DIR}/contrib/minizip")
set(include_dir "${source_dir}")

set(source_filenames "ioapi.c" "zip.c")
set(header_filenames "ioapi.h" "zip.h")

#xxx With CMake 3.12, we should be able to use the following instead:
#list(TRANSFORM ${source_filenames} PREPEND ${source_dir})
string(REGEX REPLACE "([^;]+)" "${source_dir}/\\1" source_fullpaths "${source_filenames}")
string(REGEX REPLACE "([^;]+)" "${include_dir}/\\1" header_fullpaths "${header_filenames}")

add_library("${TARGET_NAME}" ${source_fullpaths})
target_sources("${TARGET_NAME}" PUBLIC "${header_fullpaths}")
set_target_properties("${TARGET_NAME}" PROPERTIES OUTPUT_NAME "${OUTPUT_NAME}")
target_link_libraries("${TARGET_NAME}" "${MODULE_DEPS}")
target_include_directories("${TARGET_NAME}" PUBLIC
	"${include_dir}"
	$<BUILD_INTERFACE:${coda-oss_zlib_SOURCE_DIR}>
	$<BUILD_INTERFACE:${coda-oss_zlib_BINARY_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

install(TARGETS "${TARGET_NAME}"
	LIBRARY DESTINATION ${CODA_STD_PROJECT_LIB_DIR}
	ARCHIVE DESTINATION ${CODA_STD_PROJECT_LIB_DIR}
	RUNTIME DESTINATION ${CODA_STD_PROJECT_BIN_DIR}
)

install(FILES ${header_fullpaths} DESTINATION ${CODA_STD_PROJECT_INCLUDE_DIR})

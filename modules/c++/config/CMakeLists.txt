set(TARGET_NAME "config")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

set(CODA_EXPORT " ")
if (MSVC AND BUILD_SHARED_LIBS)
    set(CODA_EXPORT "__declspec(dllexport)")
endif()

set(CODA_CPP11 1)

set(config_file_out "include/config/coda_oss_config.h")
configure_file("include/config/coda_oss_config.h.in" "${config_file_out}" @ONLY)
target_include_directories("${TARGET_NAME}" INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}" DESTINATION "include/config")

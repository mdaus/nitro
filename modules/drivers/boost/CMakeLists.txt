set(TARGET_NAME "boost-config")

set(config_file_out "include/boost/boost_config.h")
configure_file("include/boost/boost_config.h.in" "${config_file_out}" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}" DESTINATION "include/boost")

add_library(${TARGET_NAME} INTERFACE)
target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CODA_STD_PROJECT_INCLUDE_DIR}>)

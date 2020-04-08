set(config_file_out "include/boost/boost_config.h")
configure_file("include/boost/boost_config.h.in" "${config_file_out}" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}"
        DESTINATION "include/boost"
        ${CODA_INSTALL_OPTION})

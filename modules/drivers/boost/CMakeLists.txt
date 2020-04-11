set(config_file_out "include/boost/boost_config.h")
configure_file("include/boost/boost_config.h.in" "${config_file_out}" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}"
        DESTINATION "include/boost"
        ${CODA_INSTALL_OPTION})
if (TARGET Boost::serialization)
    target_include_directories(Boost::serialization INTERFACE 
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>")
endif()

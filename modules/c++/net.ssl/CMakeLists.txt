set(TARGET_NAME "net.ssl")
set(MODULE_DEPS "net")

option(USE_SSL "Enable OpenSSL" ON)

if (USE_SSL)
    find_package(OpenSSL)
    if (OPENSSL_FOUND)
        coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.
        target_include_directories(${TARGET_NAME} PUBLIC "${OPENSSL_INCLUDE_DIR}")
        target_link_libraries(${TARGET_NAME} PUBLIC "${OPENSSL_SSL_LIBRARY}")

        coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
    else()
        message("OpenSSL installation not found, net.ssl will not be built")
    endif()
endif()

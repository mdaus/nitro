set(MODULE_NAME "net.ssl")

option(USE_SSL "Enable OpenSSL" ON)

if (USE_SSL)
    find_package(OpenSSL)
    if (OPENSSL_FOUND)
        coda_add_module(
            NAME ${MODULE_NAME}
            DEPS net
            EXTERNAL_DEPS ${OPENSSL_SSL_LIBRARY})
        target_include_directories(${MODULE_NAME}-c++ PUBLIC "${OPENSSL_INCLUDE_DIR}")

        coda_add_tests(
            MODULE_NAME ${MODULE_NAME}
            DIRECTORY "tests")
    else()
        message("OpenSSL installation not found, net.ssl will not be built")
    endif()
endif()

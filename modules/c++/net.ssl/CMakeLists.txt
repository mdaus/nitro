set(MODULE_NAME net.ssl)

set(SSL_HOME "" CACHE STRING "path to external libssl installation")

if (NOT SSL_HOME STREQUAL "")
    set(OPENSSL_ROOT_DIR ${SSL_HOME})
endif()
find_package(OpenSSL)
if (OPENSSL_FOUND)
    set(USE_OPENSSL 1)
    set(EXTERNAL_DEPS OpenSSL::SSL OpenSSL::Crypto)
else()
    message(WARNING "OpenSSL installation not found")
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    VERSION 1.0
    DEPS net
    EXTERNAL_DEPS ${EXTERNAL_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

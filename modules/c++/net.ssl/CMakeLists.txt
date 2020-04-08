set(MODULE_NAME net.ssl)

set(SSL_HOME "" CACHE STRING "path to external libssl installation")

if (NOT SSL_HOME STREQUAL "")
    set(OPENSSL_ROOT_DIR ${SSL_HOME})
endif()
find_package(OpenSSL)
if (OPENSSL_FOUND)
    set(USE_OPENSSL 1)
else()
    message(WARNING "OpenSSL installation not found")
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    VERSION 1.0
    DEPS net
    EXTERNAL_DEPS ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
target_include_directories(${MODULE_NAME}-c++ PUBLIC "${OPENSSL_INCLUDE_DIR}")

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

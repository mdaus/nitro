set(MODULE_NAME net.ssl)
set(MODULE_DEPS net-c++)

set(SSL_HOME "" CACHE STRING "path to external libssl installation")

if (NOT SSL_HOME STREQUAL "")
    set(OPENSSL_ROOT_DIR ${SSL_HOME})
endif()
find_package(OpenSSL)
if (OPENSSL_FOUND)
    set(USE_OPENSSL 1)
    list(APPEND MODULE_DEPS OpenSSL::SSL OpenSSL::Crypto)
else()
    message(WARNING "OpenSSL installation not found")
endif()
coda_generate_module_config_header(${MODULE_NAME})

coda_add_module(
    ${MODULE_NAME}
    VERSION 1.0
    DEPS ${MODULE_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

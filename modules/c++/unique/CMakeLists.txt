set(MODULE_NAME unique)
set(MODULE_DEPS except-c++)
# On Unix, we depend on libuuid.  On Windows, we use Rpcrt4.
if (WIN32)
    list(APPEND MODULE_DEPS Rpcrt4)
elseif (UNIX)
    list(APPEND MODULE_DEPS uuid)
endif()

coda_add_module(
    ${MODULE_NAME}
    DEPS ${MODULE_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

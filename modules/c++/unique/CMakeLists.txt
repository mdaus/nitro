set(MODULE_NAME unique)

# On Unix, we depend on libuuid.  On Windows, we use Rpcrt4.
if (WIN32)
    set(EXTERNAL_DEPS Rpcrt4)
elseif (UNIX)
    set(EXTERNAL_DEPS uuid)
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS except
    EXTERNAL_DEPS ${EXTERNAL_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

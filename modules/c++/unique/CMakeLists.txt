set(TARGET_NAME "unique")
set(MODULE_DEPS "except")

# On Unix, we depend on libuuid.  On Windows, we use Rpcrt4.
if (WIN32)
    list(APPEND MODULE_DEPS "Rpcrt4")
elseif (UNIX)
    list(APPEND MODULE_DEPS "uuid")
    set(EXTRA_DEPS "${CMAKE_PROJECT_NAME}_UUID")
endif()

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")

set(MODULE_NAME unique)
set(MODULE_DEPS except)

# On Unix, we depend on libuuid.  On Windows, we use Rpcrt4.
if (WIN32)
    set(EXTERNAL_DEPS Rpcrt4)
elseif (UNIX)
    set(EXTERNAL_DEPS uuid)
    set(EXTRA_DEPS "${CMAKE_PROJECT_NAME}_UUID")
endif()

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")

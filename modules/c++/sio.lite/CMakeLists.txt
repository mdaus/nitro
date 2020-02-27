set(TARGET_NAME "sio.lite")
set(MODULE_DEPS "io" "types")

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")

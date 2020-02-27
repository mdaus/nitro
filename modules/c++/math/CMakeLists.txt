set(TARGET_NAME "math")
set(MODULE_DEPS "except" "str" "sys" "types")

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${TARGET_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

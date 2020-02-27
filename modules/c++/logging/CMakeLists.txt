set(TARGET_NAME "logging")
set(MODULE_DEPS "io" "mt")

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${TARGET_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

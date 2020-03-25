set(MODULE_NAME str)
set(MODULE_DEPS except)
set(UNITTEST_DEPS sys)

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

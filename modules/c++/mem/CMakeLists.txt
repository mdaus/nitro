set(MODULE_NAME mem)
set(MODULE_DEPS sys)
set(TEST_DEPS cli)

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

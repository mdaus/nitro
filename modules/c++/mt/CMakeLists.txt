set(MODULE_NAME mt)
set(MODULE_DEPS sys except math mem types)
set(TEST_DEPS cli)

option(MT_DEFAULT_PINNING "Use affinity-based CPU pinning by default in MT" OFF)

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

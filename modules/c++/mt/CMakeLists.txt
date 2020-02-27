set(TARGET_NAME "mt")
set(MODULE_DEPS "sys" "except" "math" "mem" "types")
set(TEST_DEPS "cli")

option(MT_DEFAULT_PINNING "Use affinity-based CPU pinning by default in MT" OFF)

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${TARGET_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

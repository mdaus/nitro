set(TARGET_NAME "cli")
set(MODULE_DEPS "sys" "mem")

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.
coda_add_unittests(${TARGET_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

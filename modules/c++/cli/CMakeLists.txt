set(MODULE_NAME "cli")
set(MODULE_DEPS "sys" "mem")

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

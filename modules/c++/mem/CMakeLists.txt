set(TARGET_NAME "mem")
set(MODULE_DEPS "sys" "cli")
set(UNITTEST_DEPS "sys")  #xxx Ideally this would be implicit.

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

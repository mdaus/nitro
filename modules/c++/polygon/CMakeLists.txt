set(TARGET_NAME "polygon")
set(MODULE_DEPS "sys" "mem" "types" "math" "except")
set(TEST_DEPS "sio.lite")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

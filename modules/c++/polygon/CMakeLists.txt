set("TARGET_NAME"       "polygon")
set("TARGET_VERSION"    "1.0")
set("MODULE_DEPS"       "sys" "mem" "types" "math" "except")
set("TEST_DEPS"         "sio.lite")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.
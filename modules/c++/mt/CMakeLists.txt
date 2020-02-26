set(TARGET_NAME "mt")
set(MODULE_DEPS "sys" "except" "math" "mem" "types")
set(TEST_DEPS "cli")

option(MT_DEFAULT_PINNING "Use affinity-based CPU pinning by default in MT" OFF)

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

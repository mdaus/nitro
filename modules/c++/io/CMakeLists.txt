set(TARGET_NAME "io")
set(MODULE_DEPS "sys" "mem")
set(SOURCE_FILTER "MMapInputStream.cpp")
set(TEST_FILTER "mmByteStreamTest.cpp")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

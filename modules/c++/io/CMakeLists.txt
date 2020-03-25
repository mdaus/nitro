set(MODULE_NAME io)
set(MODULE_DEPS sys mem)
set(SOURCE_FILTER "MMapInputStream.cpp")
set(TEST_FILTER "mmByteStreamTest.cpp")

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

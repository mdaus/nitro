set(MODULE_NAME io)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys mem
    SOURCE_FILTER "MMapInputStream.cpp")

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    FILTER_LIST "mmByteStreamTest.cpp")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

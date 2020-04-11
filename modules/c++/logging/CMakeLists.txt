set(MODULE_NAME logging)

coda_add_module(
    ${MODULE_NAME}
    DEPS io-c++ mt-c++)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

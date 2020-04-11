set(MODULE_NAME tiff)

coda_add_module(
    ${MODULE_NAME}
    DEPS mt-c++ io-c++)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

set(MODULE_NAME tiff)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS mt io)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

set(MODULE_NAME except)

coda_add_module(
    ${MODULE_NAME}
    DEPS config-c++)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

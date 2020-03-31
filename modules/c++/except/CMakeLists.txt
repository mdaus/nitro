set(MODULE_NAME except)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS config)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

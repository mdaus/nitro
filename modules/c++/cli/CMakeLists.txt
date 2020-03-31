set(MODULE_NAME "cli")

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys mem)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

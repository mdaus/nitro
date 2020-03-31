set(MODULE_NAME mem)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    DEPS cli)
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

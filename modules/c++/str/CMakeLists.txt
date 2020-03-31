set(MODULE_NAME str)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS except)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    DEPS sys
    UNITTEST)

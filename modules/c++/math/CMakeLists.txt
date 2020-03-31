set(MODULE_NAME math)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS except str sys types)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

set(MODULE_NAME logging)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS io mt)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

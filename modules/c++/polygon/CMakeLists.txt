set(MODULE_NAME polygon)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys mem types math except)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    DEPS sio.lite)
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

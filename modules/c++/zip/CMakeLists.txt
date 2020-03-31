set(MODULE_NAME zip)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS io
    EXTERNAL_DEPS z minizip)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

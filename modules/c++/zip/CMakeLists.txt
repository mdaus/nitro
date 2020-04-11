set(MODULE_NAME zip)

coda_add_module(
    ${MODULE_NAME}
    DEPS io-c++ z minizip)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

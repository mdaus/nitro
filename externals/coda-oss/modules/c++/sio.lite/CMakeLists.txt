set(MODULE_NAME sio.lite)

coda_add_module(
    ${MODULE_NAME}
    VERSION 1.0
    DEPS sys-c++ io-c++ types-c++)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

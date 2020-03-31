set(MODULE_NAME sio.lite)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS io types)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

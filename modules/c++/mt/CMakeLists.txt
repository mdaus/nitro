set(MODULE_NAME mt)

option(MT_DEFAULT_PINNING "Use affinity-based CPU pinning by default in MT" OFF)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys except math mem types)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    DEPS cli)
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

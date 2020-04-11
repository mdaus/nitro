set(MODULE_NAME mt)

option(MT_DEFAULT_PINNING "Use affinity-based CPU pinning by default in MT" OFF)

coda_add_module(
    ${MODULE_NAME}
    DEPS sys-c++ except-c++ math-c++ mem-c++ types-c++)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    DEPS cli-c++)
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

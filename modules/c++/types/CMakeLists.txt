set(MODULE_NAME types)

coda_add_module(${MODULE_NAME})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    DEPS sys-c++
    UNITTEST)

set(MODULE_NAME sys)
set(MODULE_DEPS config-c++ except-c++ str-c++)
if (WIN32)
    list(APPEND MODULE_DEPS ws2_32)
elseif (UNIX)
    list(APPEND MODULE_DEPS pthread)
endif()
list(APPEND MODULE_DEPS "${CMAKE_DL_LIBS}" "${CLOCK_GETTIME_EXTRALIBS}")

coda_generate_module_config_header(${MODULE_NAME})

coda_add_module(
    ${MODULE_NAME}
    VERSION 1.2
    DEPS ${MODULE_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    FILTER_LIST "MMapReadOnlyTest.cpp" "ProcessTest1.cpp")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    DEPS mt-c++
    UNITTEST)

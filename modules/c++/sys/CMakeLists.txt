set(MODULE_NAME sys)

if (WIN32)
    set(EXTERNAL_DEPS ws2_32)
elseif (UNIX)
    set(EXTERNAL_DEPS pthread)
endif()
list(APPEND EXTERNAL_DEPS "${CMAKE_DL_LIBS}" "${CLOCK_GETTIME_EXTRALIBS}")

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS config except str
    EXTERNAL_DEPS ${EXTERNAL_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    FILTER_LIST "MMapReadOnlyTest.cpp" "ProcessTest1.cpp")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    DEPS mt
    UNITTEST)

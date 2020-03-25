set(MODULE_NAME sys)
set(MODULE_DEPS config except str "${CMAKE_DL_LIBS}" "${CLOCK_GETTIME_EXTRALIBS}")
set(UNITTEST_DEPS mt)
set(TEST_FILTER "MMapReadOnlyTest.cpp" "ProcessTest1.cpp")

if (WIN32)
    set(EXTERNAL_DEPS ws2_32)
elseif (UNIX)
    set(EXTERNAL_DEPS pthread)
endif()

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

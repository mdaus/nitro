set(MODULE_NAME math.poly)

if (TARGET Boost::serialization)
    set(UNITTEST_DEPS serialize-c++)
else()
    set(UNITTEST_FILTER "test_1d_poly_serialize.cpp" "test_2d_poly_serialize.cpp")
endif()

coda_add_module(
    ${MODULE_NAME}
    DEPS sys-c++ math.linear-c++)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    DEPS "${UNITTEST_DEPS}"
    FILTER_LIST "${UNITTEST_FILTER}"
    UNITTEST)

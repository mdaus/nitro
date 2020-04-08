set(MODULE_NAME math.linear)

if (TARGET Boost::serialization)
    set(UNITTEST_DEPS serialize)
else()
    set(UNITTEST_FILTER "test_Matrix2D_serialize.cpp" "test_Vector_serialize.cpp")
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys mem types)

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    DEPS "${UNITTEST_DEPS}"
    FILTER_LIST "${UNITTEST_FILTER}"
    UNITTEST)

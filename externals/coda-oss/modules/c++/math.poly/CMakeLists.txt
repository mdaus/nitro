set(MODULE_NAME math.poly)

if (Boost_serialization_FOUND)
    set(UNITTEST_DEPS serialize-c++)
else()
    set(UNITTEST_FILTER "test_1d_poly_serialize.cpp" "test_2d_poly_serialize.cpp")
endif()

coda_add_module(
    ${MODULE_NAME}
    VERSION 0.2
    DEPS sys-c++ math.linear-c++)



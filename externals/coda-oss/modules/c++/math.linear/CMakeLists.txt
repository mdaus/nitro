set(MODULE_NAME math.linear)

if (Boost_serialization_FOUND)
    set(UNITTEST_DEPS serialize-c++)
else()
    set(UNITTEST_FILTER "test_Matrix2D_serialize.cpp" "test_Vector_serialize.cpp")
endif()

coda_add_module(
    ${MODULE_NAME}
    VERSION 0.2
    DEPS sys-c++ mem-c++ types-c++)



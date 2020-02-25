set(TARGET_NAME "math.poly")
set(MODULE_DEPS "sys" "math.linear")
set(TEST_DEPS ${MODULE_DEPS}) # TODO: fix transitive dependency resolution
set(UNITTEST_DEPS ${MODULE_DEPS}) # TODO: fix transitive dependency resolution
if (Boost_serialization_FOUND)
    list(APPEND UNITTEST_DEPS "serialize")
else()
    set(UNITTEST_FILTER "test_1d_poly_serialize.cpp" "test_2d_poly_serialize.cpp")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

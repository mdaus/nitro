set(TARGET_NAME "math.poly")
set(MODULE_DEPS "sys" "math.linear")

if (HAVE_BOOST) #xxx Review this test
    set(UNITTEST_DEPS "serialize")
    set(UNITTEST_USELIB "BOOST_SERIALIZATION")
else()
    set(UNITTEST_FILTER "test_1d_poly_serialize.cpp" "test_2d_poly_serialize.cpp")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

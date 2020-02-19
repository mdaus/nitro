set(TARGET_NAME "math.linear")
set(MODULE_DEPS "sys" "mem" "types")

if (HAVE_BOOST) #xxx Review this test
    set(UNITTEST_DEPS "serialize")
    set(UNITTEST_USELIB "BOOST_SERIALIZATION")
else()
    set(UNITTEST_FILTER "test_Matrix2D_serialize.cpp" "test_Vector_serialize.cpp")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

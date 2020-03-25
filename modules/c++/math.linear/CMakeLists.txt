set(MODULE_NAME math.linear)
set(MODULE_DEPS sys mem types)

if (TARGET serialize)
    set(UNITTEST_DEPS serialize)
else()
    set(UNITTEST_FILTER "test_Matrix2D_serialize.cpp" "test_Vector_serialize.cpp")
endif()

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

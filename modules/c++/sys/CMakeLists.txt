set(TARGET_NAME "sys")
set(MODULE_DEPS "config" "except" "str" "${CMAKE_DL_LIBS}" "${CLOCK_GETTIME_EXTRALIBS}")
set(UNITTEST_DEPS "mt")
set(TEST_FILTER "MMapReadOnlyTest.cpp" "ProcessTest1.cpp")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (WIN32)
    target_link_libraries("${TARGET_NAME}" ws2_32)
elseif (UNIX)
    target_link_libraries("${TARGET_NAME}" pthread)
endif()

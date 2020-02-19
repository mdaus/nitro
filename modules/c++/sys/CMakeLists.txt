set(TARGET_NAME "sys")
set(USELIB "THREAD" "DL" "RT" "SOCKET")
set(MODULE_DEPS "config" "except" "str" "Threads::Threads" "${CMAKE_DL_LIBS}" "${CLOCK_GETTIME_EXTRALIBS}")
set(UNITTEST_DEPS "mt")
set(TEST_FILTER "MMapReadOnlyTest.cpp" "ProcessTest1.cpp")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (WIN32) #xxx Adapt for X-platform
    target_link_libraries("${TARGET_NAME}" ws2_32)
endif()

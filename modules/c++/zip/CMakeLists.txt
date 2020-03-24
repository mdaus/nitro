set(TARGET_NAME "zip")
set(MODULE_DEPS "io" ${LIBZ_LIBRARY} "${CMAKE_PROJECT_NAME}_minizip")

coda_add_library(${TARGET_NAME})  # See the definition of this method for details on the variables it looks at.

#xxx If we want to use FindZLib instead:
#set("MODULE_DEPS" "io" "ZLIB::ZLIB" "${CMAKE_PROJECT_NAME}_minizip")
#if (ZLIB_FOUND)
#    target_include_directories(${TARGET_NAME} ${ZLIB_INCLUDE_DIRS})
#    target_link_libraries(${TARGET_NAME} ${ZLIB_LIBRARIES})
#endif()

coda_add_tests(${TARGET_NAME} "${TEST_DEPS}" "${TEST_FILTER}")

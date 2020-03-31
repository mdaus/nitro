set(MODULE_NAME zip)

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS io
    EXTERNAL_DEPS ${LIBZ_LIBRARY} "${CMAKE_PROJECT_NAME}_minizip")

#xxx If we want to use FindZLib instead:
#set("MODULE_DEPS" "io" "ZLIB::ZLIB" "${CMAKE_PROJECT_NAME}_minizip")
#if (ZLIB_FOUND)
#    target_include_directories(${MODULE_NAME} ${ZLIB_INCLUDE_DIRS})
#    target_link_libraries(${MODULE_NAME} ${ZLIB_LIBRARIES})
#endif()

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")

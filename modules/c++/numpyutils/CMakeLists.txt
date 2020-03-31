set(MODULE_NAME numpyutils)

if(Python_FOUND AND Python_Development_FOUND AND Python_NumPy_FOUND)
    coda_add_module(
        NAME ${MODULE_NAME}
        DEPS types except sys
        EXTERNAL_DEPS ${Python_LIBRARIES})
    target_include_directories(${MODULE_NAME}-c++ PUBLIC
        ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

    coda_add_tests(
        MODULE_NAME ${MODULE_NAME}
        DIRECTORY "unittests"
        UNITTEST)
endif()

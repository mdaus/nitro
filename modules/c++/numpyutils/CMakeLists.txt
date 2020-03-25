set(MODULE_NAME numpyutils)
set(MODULE_DEPS types except sys)

if(Python_FOUND AND Python_Development_FOUND AND Python_NumPy_FOUND)
    set(EXTERNAL_DEPS ${Python_LIBRARIES})
    coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.
    target_include_directories(${MODULE_NAME}-c++ PUBLIC
        ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

    coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")
endif()

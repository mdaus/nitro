set(TARGET_NAME "numpyutils")
set(MODULE_DEPS "types" "except" "sys")

if(Python_FOUND AND Python_Development_FOUND AND Python_NumPy_FOUND)
    coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.
    target_include_directories("${TARGET_NAME}"
        PUBLIC ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
    target_link_libraries("${TARGET_NAME}" PUBLIC ${Python_LIBRARIES})
endif()

set(TARGET_NAME "numpyutils")
set(MODULE_DEPS "types" "except" "sys")

if (PYTHON_HOME)
    set(Python_ROOT_DIR ${PYTHON_HOME})
endif()

find_package(Python COMPONENTS Interpreter Development NumPy)

if(Python_FOUND)
    coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.
    target_include_directories("${TARGET_NAME}"
        PUBLIC ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
    target_link_directories("${TARGET_NAME}" PUBLIC ${Python_LIBRARY_DIRS})
    if(NOT PYTHON_HOME)
        message("Python installation found at ${Python_EXECUTABLE}.\n"
                "Pass the configure option -DPYTHON_HOME=... to override this selection.")
    endif()
else()
    message("${TARGET_NAME} will not be built since Python libraries were not found.\n"
            "Pass the configure option -DPYTHON_HOME=... to help locate an installation.")
endif()

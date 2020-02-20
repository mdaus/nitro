set(TARGET_LANG "python")


if (PYTHON_HOME)
    set(Python_ROOT_DIR ${PYTHON_HOME})
endif()

find_package(SWIG)
find_package(Python COMPONENTS Interpreter Development)

if(SWIG_FOUND AND Python_FOUND)
    add_subdirectory("except")
    #add_subdirectory("sys")
    #add_subdirectory("types")
    #add_subdirectory("mem")
    #add_subdirectory("mt")
    #add_subdirectory("io")
    #add_subdirectory("logging")
    #add_subdirectory("sio.lite")

    #add_subdirectory("math.linear")
    #add_subdirectory("math.poly")
    #add_subdirectory("xml.lite")
endif()

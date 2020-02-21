set(TARGET_LANG "python")

if (PYTHON_HOME)
    set(Python_ROOT_DIR ${PYTHON_HOME})
endif()

find_package(SWIG)
find_package(Python COMPONENTS Interpreter Development)
if(SWIG_FOUND AND Python_FOUND)
    cmake_policy(SET CMP0078 NEW) # UseSWIG generates standard target names
    cmake_policy(SET CMP0086 NEW) # UseSWIG honors SWIG_MODULE_NAME via -module
    include(${SWIG_USE_FILE})
    option(PYTHON_EXTRA_NATIVE "generate extra native containers with SWIG" ON)
    option(PYTHON_AGNOSTIC "generate Python 2/3 agnostic bindings" OFF)

    set(CODA_PYTHON_SITE_PACKAGES
        "${CODA_STD_PROJECT_LIB_DIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")

    set(CMAKE_SWIG_FLAGS "")
    if (PYTHON_EXTRA_NATIVE)
        list(APPEND CMAKE_SWIG_FLAGS "-extranative")
    endif()
    if (NOT PYTHON_AGNOSTIC)
        list(APPEND CMAKE_SWIG_FLAGS "-py3")
    endif()

    add_subdirectory("except")
    add_subdirectory("sys")
    #add_subdirectory("types")
    #add_subdirectory("mem")
    #add_subdirectory("mt")
    add_subdirectory("io")
    add_subdirectory("logging")
    #add_subdirectory("sio.lite")

    #add_subdirectory("math.linear")
    #add_subdirectory("math.poly")
    #add_subdirectory("xml.lite")
endif()

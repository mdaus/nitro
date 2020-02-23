# - CODA-OSS CMake Build script.
#
# Author:  Scott A. Colcord

# We use FetchContent, which requires 3.11, in the zlib build
cmake_minimum_required(VERSION 3.11)

project("coda-oss")

set(CMAKE_CXX_STANDARD 11)

# Standard directory names used throughout the project.
#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR/target}) # Specified in CMakeSettings.json
set(CODA_STD_BUILDUTIL_DIR          "build")
set(CODA_STD_BUILD_DIR              "target")
set(CODA_STD_INSTALL_DIR            "install")
set(CODA_STD_PROJECT_SOURCE_DIR     "source")
set(CODA_STD_PROJECT_INCLUDE_DIR    "include")
set(CODA_STD_PROJECT_IMPORT_DIR     "import")
set(CODA_STD_PROJECT_LIB_DIR        "lib")
set(CODA_STD_PROJECT_BIN_DIR        "bin")
set(CODA_STD_PROJECT_TESTS_DIR      "tests")
set(CODA_STD_PROJECT_UNITTESTS_DIR  "unittests")

# Detect 32/64-bit architecture
#xxx Might still need to set -m32 or -m64 compiler and linker flags if not done automatically
if(NOT CODA_BUILD_BITSIZE)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CODA_BUILD_BITSIZE "64" CACHE STRING "Select Architecture" FORCE)
    elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(CODA_BUILD_BITSIZE "32" CACHE STRING "Select Architecture" FORCE)
    else()
        message(FATAL_ERROR "Unknown Pointer Size: ${CMAKE_SIZEOF_VOID_P} Bytes")
    endif()
    set_property(CACHE CODA_BUILD_BITSIZE PROPERTY STRINGS "64" "32")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Select Build Type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build shared libraries instead of static." OFF)
if(BUILD_SHARED_LIBS)
    set(CODA_LIBRARY_TYPE "shared")
else()
    set(CODA_LIBRARY_TYPE "static")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_BUILDUTIL_DIR}")

# Set default install directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message("Overriding default CMAKE_INSTALL_PREFIX of ${CMAKE_INSTALL_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "${CODA_STD_INSTALL_DIR}/${CMAKE_SYSTEM_NAME}${CODA_BUILD_BITSIZE}-${CMAKE_BUILD_TYPE}-${CODA_LIBRARY_TYPE}" CACHE PATH "Install directory" FORCE)
endif()

# Look for things in our own install location first.
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}")

include(config_tests) # Code to test compiler features

option(CODA_BUILD_TESTS "build tests" ON)

# MSVC-specific flags and options.
if (MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Remove any default settings that we don't want.
    string(REGEX REPLACE "/W[0-3]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "/W[0-3]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    add_definitions(
        -DWIN32_LEAN_AND_MEAN
        -DNOMINMAX
        -D_CRT_SECURE_NO_WARNINGS
        -D_SCL_SECURE_NO_WARNINGS
        -D_USE_MATH_DEFINES
    )
    add_compile_options(
        /wd4290
        /wd4512
        /EHs        # Needed when exceptions might bubble through a C-linkage layer
#       /MP         # Parallel Build
    )

    link_libraries(    # CMake uses this for both libraries and linker options
        -STACK:80000000
    )

    # This should probably be replaced by GenerateExportHeader
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
endif()

# Unix/Linux specific options
if (UNIX)
    add_definitions(
        -D_LARGEFILE_SOURCE
        -D_FILE_OFFSET_BITS=64
    )
    add_compile_options(
        -Wno-deprecated
        -Wno-unused-value
        -Wno-unused-but-set-variable
        -Wno-misleading-indentation
    )
endif()

include(CodaBuild.cmake)

find_package(SWIG)
if (SWIG_FOUND)
    cmake_policy(SET CMP0078 NEW) # UseSWIG generates standard target names
    cmake_policy(SET CMP0086 NEW) # UseSWIG honors SWIG_MODULE_NAME via -module
    include(${SWIG_USE_FILE})
    option(PYTHON_EXTRA_NATIVE "generate extra native containers with SWIG" ON)

    set(CMAKE_SWIG_FLAGS "")
    if (PYTHON_EXTRA_NATIVE)
        list(APPEND CMAKE_SWIG_FLAGS "-extranative")
    endif()
    list(APPEND CMAKE_SWIG_FLAGS "-py3")
else()
    message(WARNING "SWIG could not be found, so Python modules relying on it will not be built")
endif()

if (PYTHON_HOME)
    set(Python_ROOT_DIR ${PYTHON_HOME})
endif()
find_package(Python COMPONENTS Interpreter Development NumPy)
if (Python_FOUND AND Python_Development_FOUND)
    set(CODA_PYTHON_SITE_PACKAGES
        "${CODA_STD_PROJECT_LIB_DIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    if(NOT PYTHON_HOME)
        message("Python installation found at ${Python_EXECUTABLE}.\n"
                "Pass the configure option -DPYTHON_HOME=... to override this selection.")
    endif()
else()
    message(WARNING "Python targets will not be built since Python libraries were not found.\n"
            "Pass the configure option -DPYTHON_HOME=... to help locate an installation.")
endif()

add_subdirectory("modules/drivers")
enable_testing()
add_subdirectory("modules/c++")
if (SWIG_FOUND AND Python_FOUND AND Python_Development_FOUND)
    add_subdirectory("modules/python")
endif()

# - CODA-OSS CMake Build script.
#
# Author:  Scott A. Colcord

# We use FetchContent, which requires 3.11, in the zlib build
cmake_minimum_required(VERSION 3.11)

project("coda-oss")

set(CMAKE_CXX_STANDARD 11)

if (${CMAKE_PROJECT_NAME} STREQUAL coda-oss)
    include(CodaBuild.cmake)
endif()

add_subdirectory("modules/drivers")

# let internal modules find the installed external dependencies
link_directories(${CMAKE_INSTALL_PREFIX}/${CODA_STD_PROJECT_LIB_DIR})

add_subdirectory("modules/c++")

if (SWIG_FOUND AND Python_FOUND AND Python_Development_FOUND)
    add_subdirectory("modules/python")
endif()

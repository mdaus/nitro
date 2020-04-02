# - CODA-OSS CMake Build script.
#
# Author:  Scott A. Colcord

cmake_minimum_required(VERSION 3.11)

project("coda-oss")

set(CMAKE_CXX_STANDARD 11)

if (${CMAKE_PROJECT_NAME} STREQUAL coda-oss)
    include(CodaBuild.cmake)
    coda_initialize_build()
endif()

add_subdirectory("modules/drivers")
add_subdirectory("modules/c++")
if (SWIG_FOUND AND Python_FOUND AND Python_Development_FOUND)
    add_subdirectory("modules/python")
endif()

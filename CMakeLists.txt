# - CODA-OSS CMake Build script.
#
# Author:  Scott A. Colcord

cmake_minimum_required(VERSION 3.14)

project(coda-oss)

set(CMAKE_CXX_STANDARD 11)

if (${CMAKE_PROJECT_NAME} STREQUAL coda-oss)
    include(cmake/CodaBuild.cmake)
    coda_initialize_build()

    # install our cmake build modules for downstream use
    install(DIRECTORY "cmake/"
            DESTINATION "${CODA_STD_PROJECT_LIB_DIR}/cmake/"
            FILES_MATCHING PATTERN "*.cmake")
endif()

add_subdirectory("modules")

coda_generate_package_config()

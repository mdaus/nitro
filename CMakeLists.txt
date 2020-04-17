cmake_minimum_required(VERSION 3.14)
project(nitro)

set(CMAKE_C_STANDARD 90)
set(CMAKE_CXX_STANDARD 11)

if (${CMAKE_PROJECT_NAME} STREQUAL nitro)
    # this is the top-level project
    set(CODA_OSS_HOME "" CACHE PATH "location of coda-oss install, if not provided we will build it")

    if (EXISTS ${CMAKE_BINARY_DIR}/conan_paths.cmake)
        # use pre-built coda-oss from conan package
        include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
        find_package(coda-oss REQUIRED)
        include(CodaBuild)
        coda_initialize_build()
    elseif (CODA_OSS_HOME)
        # use pre-built coda-oss from specified install location
        list(APPEND CMAKE_PREFIX_PATH "${CODA_OSS_HOME}/lib/cmake")
        find_package(coda-oss)
        include(CodaBuild)
        coda_initialize_build()
    else()
        # build coda-oss ourselves
        include("externals/coda-oss/cmake/CodaBuild.cmake")
        coda_initialize_build()
        add_subdirectory("externals/coda-oss")
    endif()
else()
    # don't initialize or add externals, the top-level project is responsible for that
endif()

set(NITRO_C_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/c)
add_subdirectory("modules")

coda_generate_package_config()

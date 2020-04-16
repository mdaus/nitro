# CODA_OSS uses FetchContent, which requires 3.11, in the zlib build
#   It might be possible to rework if this has to work with older CMakes
cmake_minimum_required(VERSION 3.11)

project(nitro)

include(common_build.cmake)

list(APPEND coda_modules except-c++ io-c++ sys-c++ mt-c++ str-c++
                         math-c++ xml.lite-c++ jpeg J2K)

if (${CMAKE_PROJECT_NAME} STREQUAL nitro)
    # this is the top-level project
    set(CODA_OSS_HOME "" CACHE PATH "location of coda-oss install, if not provided we will build it")

    if (EXISTS ${CMAKE_BINARY_DIR}/conan_paths.cmake)
        # use pre-built coda-oss from conan package
        include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
        find_package(coda-oss)
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
        include(externals/coda-oss/cmake/CodaBuild.cmake)
        coda_initialize_build()
        add_subdirectory("externals/coda-oss")
    endif()
else()
    # don't initialize or add externals, the top-level project is responsible for that
endif()

# if (NOT DEFINED CODA_INSTALL_DIR)
#     include(CMakeLists-CODA.txt)
# else()
#     add_library(coda-oss INTERFACE)
# endif()

# foreach(module ${coda_modules})
#     import_coda_module(${module})
# endforeach()

# add_library(coda_oss_config INTERFACE)
# target_include_directories(coda_oss_config INTERFACE ${CODA_DIR}/target/moudules/c++/config/include/)
# add_dependencies(coda_oss_config coda-oss)

# Everyone can include coda's headers
include_directories(${CODA_INSTALL_DIR}/include)

enable_testing()
set(NITRO_C_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules/c)
add_subdirectory(modules/c)
add_subdirectory(modules/c++)
if (NOT MSVC)
    add_subdirectory(modules/python)
endif()
add_subdirectory(modules/java)
add_subdirectory(modules/mex)

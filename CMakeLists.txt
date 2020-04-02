# CODA_OSS uses FetchContent, which requires 3.11, in the zlib build
#   It might be possible to rework if this has to work with older CMakes
cmake_minimum_required(VERSION 3.11)

project("nitro")

include(common_build.cmake)

if(PYTHON_HOME)
    set(Python_ROOT_DIR ${PYTHON_HOME})
endif()

find_package(Python COMPONENTS Interpreter Development NumPy)
list(APPEND coda_modules except-c++ io-c++ sys-c++ mt-c++ str-c++
                         math-c++ xml.lite-c++ jpeg J2K)

if (Python_NumPy_FOUND AND NOT MSVC)
    list(APPEND coda_modules numpyutils-c++)
endif()
if (Python_FOUND)
    set(PYTHON ${Python_EXECUTABLE})
else()
    set(PYTHON python)
endif()


if (${CMAKE_PROJECT_NAME} STREQUAL nitro)
    include(externals/coda-oss/CodaBuild.cmake)
    coda_initialize_build()
endif()
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    # this is the top-level project
    add_subdirectory(externals/coda-oss)
else()
    # don't add externals, the top-level project is responsible for that
endif()
add_library(coda-oss INTERFACE)
add_library(coda_oss_config ALIAS config-c++)

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

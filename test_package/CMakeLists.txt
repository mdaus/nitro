cmake_minimum_required(VERSION 3.14)
project(nitro-test_package)
set(CMAKE_CXX_STANDARD 11)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
find_package(nitro)

add_executable(test test.cpp)
target_link_libraries(test nitf-c++)

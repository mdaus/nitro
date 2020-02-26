set(TARGET_NAME "serialize")
set(MODULE_DEPS "boost-config")

if (${Boost_serialization_FOUND})
    coda_add_library(${TARGET_NAME})
    target_include_directories(${TARGET_NAME} INTERFACE ${Boost_INCLUDE_DIRS})
endif()

set(TARGET_NAME "serialize")
set(MODULE_DEPS "boost-config")

if (Boost_SERIALIZATION_FOUND)
    coda_add_library(${TARGET_NAME})
endif()

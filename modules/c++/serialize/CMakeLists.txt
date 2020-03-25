set(MODULE_NAME serialize)
set(EXTERNAL_DEPS boost-config)

if (Boost_SERIALIZATION_FOUND)
    coda_add_library(${MODULE_NAME})
endif()

if (Boost_SERIALIZATION_FOUND)
    coda_add_module(
        NAME serialize
        EXTERNAL_DEPS boost-config)
endif()

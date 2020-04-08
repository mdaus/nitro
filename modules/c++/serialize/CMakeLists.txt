if (TARGET Boost::serialization)
    coda_add_module(
        NAME serialize
        EXTERNAL_DEPS Boost::serialization)
endif()

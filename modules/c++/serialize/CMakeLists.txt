if (TARGET Boost::serialization)
    coda_add_module(
        serialize
        VERSION 1.0
        DEPS Boost::serialization)
endif()

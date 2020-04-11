if (TARGET Boost::serialization)
    coda_add_module(
        serialize
        DEPS Boost::serialization)
endif()

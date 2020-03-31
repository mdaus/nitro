set(MODULE_NAME net)

if (CURL_FOUND)
    set(EXTERNAL_DEPS "CURL::libcurl")  # From FindCURL
    set(NET_CURL_SUPPORT "1")
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS logging re io mem
    EXTERNAL_DEPS ${EXTERNAL_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    FILTER_LIST "AckMulticastSender.cpp" "AckMulticastSubscriber.cpp"
                "MulticastSender.cpp" "MulticastSubscriber.cpp"
                "SerializableTestClient.cpp")

set(MODULE_NAME net)
set(MODULE_DEPS logging re io mem)
set(TEST_FILTER "AckMulticastSender.cpp" "AckMulticastSubscriber.cpp" "MulticastSender.cpp"
                "MulticastSubscriber.cpp" "SerializableTestClient.cpp")

if (CURL_FOUND)
    set(EXTERNAL_DEPS "CURL::libcurl")  # From FindCURL
    set(NET_CURL_SUPPORT "1")
endif()

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")

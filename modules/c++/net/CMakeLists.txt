set(TARGET_NAME "net")
set(MODULE_DEPS "logging" "re" "io" "mem")
#xxx set(USELIB "NSL" "SOCKET")
#xxx set(USE "CURL")
set("TEST_FILTER" "AckMulticastSender.cpp" "AckMulticastSubscriber.cpp" "MulticastSender.cpp"
                  "MulticastSubscriber.cpp" "SerializableTestClient.cpp")

if (CURL_FOUND)
    set("MODULE_DEPS" "${MODULE_DEPS}" "CURL::libcurl")  # From FindCURL
    set("NET_CURL_SUPPORT" "1")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

set("TARGET_NAME"		"net")
set("MAINTAINER"		"jmrandol@users.sourceforge.net")
set("TARGET_VERSION"	"1.0")
set("MODULE_DEPS"		"logging" "re" "io" "mem")
#xxx set("USELIB"			"NSL" "SOCKET")
#xxx set("USE"				"CURL")
set("TEST_FILTER"		"AckMulticastSender.cpp" "AckMulticastSubscriber.cpp" "MulticastSender.cpp" 
						"MulticastSubscriber.cpp" "SerializableTestClient.cpp")

if (HAVE_CURL)
	set("NET_CURL_SUPPORT" "1")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.
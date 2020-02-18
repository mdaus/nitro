set("TARGET_NAME"		"xml.lite")
set("TARGET_VERSION"	"1.2")
set("MODULE_DEPS"		"io" "mem" "mt" "logging")
set("UNITTEST_DEPS"		"sys" "xerces-c")
set("TEST_FILTER"		"MMParserTest1.cpp" "MinidomParserTest2.cpp")
set("TEST_DEPS"			"cli" "xerces-c")

set("CODA_XML_LAYER" "XercesC")
set("USE_${CODA_XML_LAYER}"	TRUE)

if (USE_XercesC)
	list(APPEND "MODULE_DEPS" "xerces-c")
	coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.
endif()

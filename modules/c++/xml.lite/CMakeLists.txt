set(TARGET_NAME "xml.lite")
set(MODULE_DEPS "io" "mem" "mt" "logging")
set(UNITTEST_DEPS "sys")
set(TEST_FILTER "MMParserTest1.cpp" "MinidomParserTest2.cpp")
set(TEST_DEPS "cli")

set(CODA_XML_LAYER "XercesC")
set("USE_${CODA_XML_LAYER}" TRUE)

if (USE_XercesC)
    set(EXTRA_DEPS "coda-oss_XercesC")
    list(APPEND MODULE_DEPS "xerces-c")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

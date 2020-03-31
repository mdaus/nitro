set(MODULE_NAME xml.lite)

if (USE_XercesC)
    set(EXTERNAL_DEPS xerces-c)
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS io mem mt logging
    EXTERNAL_DEPS ${EXTERNAL_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    DEPS cli
    FILTER_LIST "MMParserTest1.cpp" "MinidomParserTest2.cpp")

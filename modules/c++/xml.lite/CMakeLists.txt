set(MODULE_NAME xml.lite)
set(MODULE_DEPS io-c++ mem-c++ mt-c++ logging-c++)
if (USE_XercesC)
    list(APPEND MODULE_DEPS xerces-c)
endif()

coda_add_module(
    ${MODULE_NAME}
    DEPS ${MODULE_DEPS})

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests"
    DEPS cli-c++
    FILTER_LIST "MMParserTest1.cpp" "MinidomParserTest2.cpp")

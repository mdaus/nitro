set(MODULE_NAME re)

# Enable to use std::regex instead of PCRE.
option(RE_ENABLE_STD_REGEX "use std::regex instead of pcre" OFF)

if (NOT RE_ENABLE_STD_REGEX)
    set(EXTERNAL_DEPS pcre2)
endif()

coda_add_module(
    NAME ${MODULE_NAME}
    DEPS sys
    EXTERNAL_DEPS ${EXTERNAL_DEPS})

if (NOT RE_ENABLE_STD_REGEX)
    if (NOT BUILD_SHARED_LIBS)
        # this definition is required to statically link against pcre2
        # see NON_AUTOTOOLS_BUILD, LINKING PROGRAMS IN WINDOWS ENVIRONMENTS section
        target_compile_definitions(${MODULE_NAME}-c++ PUBLIC -DPCRE2_STATIC)
    endif()
endif()

coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "tests")
coda_add_tests(
    MODULE_NAME ${MODULE_NAME}
    DIRECTORY "unittests"
    UNITTEST)

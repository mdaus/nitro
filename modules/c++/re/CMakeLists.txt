set(MODULE_NAME re)
set(MODULE_DEPS sys)

# Enable to use std::regex instead of PCRE.
option(RE_ENABLE_STD_REGEX "use std::regex instead of pcre" OFF)

if (NOT RE_ENABLE_STD_REGEX)
    set(EXTERNAL_DEPS pcre2)
endif()

coda_add_library(${MODULE_NAME})  # See the definition of this method for details on the variables it looks at.

if (NOT RE_ENABLE_STD_REGEX)
    if (NOT BUILD_SHARED_LIBS)
        # this definition is required to statically link against pcre2
        # see NON_AUTOTOOLS_BUILD, LINKING PROGRAMS IN WINDOWS ENVIRONMENTS section
        target_compile_definitions(${MODULE_NAME}-c++ PUBLIC -DPCRE2_STATIC)
    endif()
endif()

coda_add_tests(${MODULE_NAME} "${TEST_DEPS}" "${TEST_FILTER}")
coda_add_unittests(${MODULE_NAME} "${UNITTEST_DEPS}" "${UNITTEST_FILTER}")

set(TARGET_NAME "re")
set(MODULE_DEPS "sys")

# Enable to use std::regex instead of PCRE.
option(RE_ENABLE_STD_REGEX "use std::regex instead of pcre" OFF)

if (NOT RE_ENABLE_STD_REGEX)
    set(EXTRA_DEPS "coda-oss_PCRE")
    set(MODULE_DEPS "${MODULE_DEPS}" "pcre2-8")
    #set(TEST_DEPS "pcre2-8")
    #set(UNITTEST_DEPS "pcre2-8")
endif()

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (NOT RE_ENABLE_STD_REGEX)
    if (NOT BUILD_SHARED_LIBS)
        # this definition is required to statically link against pcre2
        # see NON_AUTOTOOLS_BUILD, LINKING PROGRAMS IN WINDOWS ENVIRONMENTS section
        target_compile_definitions(${TARGET_NAME} PUBLIC -DPCRE2_STATIC)
    endif()
endif()

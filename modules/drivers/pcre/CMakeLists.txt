set(TARGET_NAME "PCRE")

option(PCRE2_BUILD_PCRE2GREP "build pcre2grep utility" OFF)
option(PCRE2_BUILD_TESTS "build pcre2 tests" OFF)
option(PCRE2_SUPPORT_LIBBZ2 "enable pcre2 support for libbz2" OFF)
option(PCRE2_SUPPORT_LIBZ "enable pcre2 support for libz" OFF)
option(PCRE2_SUPPORT_LIBEDIT "enable pcre2 support for libedit" OFF)
option(PCRE2_SUPPORT_LIBREADLINE "enable pcre2 support for libreadline" OFF)

coda_add_driver("${TARGET_NAME}"
    "pcre2-10.34.tar.gz"
    "SHA256=DA6ABA7BA2509E918E41F4F744A59FA41A2425C59A298A232E7FE85691E00379"
    TRUE
)

# Note that PCRE doesn't export targets for external use.  So dependent projects
#  will need to reference the PCRE targets and include directores manually.
# Notable targets from the PCRE CMake scripts include:
#  pcre2-8 pcre2-32 : 8-bit and 32-bit targets
# ${PCRE2_BINARY_DIR} : Where pcre2.h will be generated.

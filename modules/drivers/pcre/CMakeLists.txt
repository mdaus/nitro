set(TARGET_NAME PCRE)

set(EXTRA_CMAKE_ARGS
    -DPCRE2_BUILD_PCRE2GREP=OFF
    -DPCRE2_BUILD_TESTS=OFF
    -DPCRE2_SUPPORT_LIBBZ2=OFF
    -DPCRE2_SUPPORT_LIBZ=OFF
    -DPCRE2_SUPPORT_LIBEDIT=OFF
    -DPCRE2_SUPPORT_LIBREADLINE=OFF)

coda_add_driver(${TARGET_NAME}
    "pcre2-10.34.tar.gz"
    "SHA256=DA6ABA7BA2509E918E41F4F744A59FA41A2425C59A298A232E7FE85691E00379"
)

# Note that PCRE doesn't export targets for external use.  So dependent projects
#  will need to reference the PCRE targets and include directores manually.
# Notable targets from the PCRE CMake scripts include:
#  pcre2-8 pcre2-32 : 8-bit and 32-bit targets
# ${PCRE2_BINARY_DIR} : Where pcre2.h will be generated.

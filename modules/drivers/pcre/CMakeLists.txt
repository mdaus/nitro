set("TARGET_NAME"       "PCRE")

# Build as a completely external target
# Technique 3a: Import PCRE's whole build system here.
# set(PCRE2_BUILD_PCRE2_32 TRUE) # Uncomment if we want the 32-bit version

set(PCRE2_BUILD_PCRE2GREP OFF)
set(PCRE2_BUILD_TESTS OFF)
set(PCRE2_SUPPORT_LIBBZ2 OFF)
set(PCRE2_SUPPORT_LIBZ OFF)
set(PCRE2_SUPPORT_LIBEDIT OFF)
set(PCRE2_SUPPORT_LIBREADLINE OFF)
coda_add_driver("${TARGET_NAME}" 
    "pcre2-10.23.tar.gz" "SHA256=bd62f955146a41815d6e2ce011bb75740825c32906b06e377f478599f54ee0dd"
    TRUE
)
# Note that PCRE doesn't export targets for external use.  So dependent projects
#  will need to reference the PCRE targets and include directores manually.
# Notable targets from the PCRE CMake scripts include:
#  pcre2-8 pcre2-32 : 8-bit and 32-bit targets
# ${PCRE2_BINARY_DIR} : Where pcre2.h will be generated.

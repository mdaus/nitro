set("TARGET_NAME"       "PCRE")

# Build as a completely external target
# Technique 3a: Import PCRE's whole build system here.
# set(PCRE2_BUILD_PCRE2_32 TRUE) # Uncomment if we want the 32-bit version
coda_add_driver("${TARGET_NAME}" 
	"pcre2-10.22.tar.gz" "SHA256=6677c66014eba5610b8fbd8196fd4e5904db4116cff1b5a0d14e91121eda4377"
	TRUE
)
# Note that PCRE doesn't export targets for external use.  So dependent projects
#  will need to reference the PCRE targets and include directores manually.
# Notable targets from the PCRE CMake scripts include:
#  pcre2-8 pcre2-32 : 8-bit and 32-bit targets
# ${PCRE2_BINARY_DIR} : Where pcre2.h will be generated.

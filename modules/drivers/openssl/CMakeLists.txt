include(ExternalProject) # Requires CMake 3.11+

#xxx ExternalProject may not work here if we want net.ssl to use it in the same build launch.

# Build as external project
#xxx This could be enhanced to optionally fetch from the official download site.
# Note that OpenSSL uses a custom build system, which only supports building within its source tree.
#xxx Also, note that openssl 0.9.8 has a build bug (#2094) on Linux64, which we work around 
#    by using the 'no-asm' flag.  This bug was fixed in 2010, so this flag can be removed when
#    we upgrade.
#xxx Finally, note that this build fails when run in parallel, so we override
#    the make flags to force a serial build.
ExternalProject_Add("${CMAKE_PROJECT_NAME}_OPENSSL"
	URL "${CMAKE_CURRENT_SOURCE_DIR}/openssl-0.9.8.tar"
	URL_HASH SHA256=2aa8d603a4e4ae441417afa37bd81db2d9cf951bcfee5895b8071f558c5f9337
	INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
	CONFIGURE_COMMAND sh <SOURCE_DIR>/config no-asm --install-prefix=<INSTALL_DIR>
	BUILD_IN_SOURCE true
	BUILD_COMMAND $(MAKE)
	INSTALL_COMMAND $(MAKE) install
)
#xxx Maybe change build directory of above to match that of other projects?

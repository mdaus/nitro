set("TARGET_NAME"		"unique")
set("MAINTAINER"		"jmrandol@users.sourceforge.net")
set("TARGET_VERSION"	"1.0")
set("MODULE_DEPS"		"except")
#[[xxx
USELIB          = 'RPC'
USELIB_CHECK    = 'UUID'
TEST_USELIB     = 'RPC'
]]

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

if (WIN32) #xxx Adapt for X-platform
	target_link_libraries("${TARGET_NAME}" Rpcrt4)
endif()
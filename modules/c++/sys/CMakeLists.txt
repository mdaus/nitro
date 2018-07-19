set("TARGET_NAME"		"sys")
set("MAINTAINER"		"jmrandol@users.sourceforge.net")
set("TARGET_VERSION"	"1.2")
set("USELIB"			"THREAD" "DL" "RT" "SOCKET")
set("MODULE_DEPS"		"except" "str") 
set("UNITTEST_DEPS"		"mt")
set("TEST_FILTER"		"MMapReadOnlyTest.cpp" "ProcessTest1.cpp")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

#xxx This bit of code from the wscript has yet to be adapted to CMake.
# this check defines HAVE_CLOCK_GETTIME, undefine it to keep it out of dumpenv
#if conf.check_cc(lib='rt', uselib_store='RT', function_name='clock_gettime', header_name='time.h', mandatory=False):
#	conf.undefine('HAVE_CLOCK_GETTIME')


if (WIN32) #xxx Adapt for X-platform
	target_link_libraries("${TARGET_NAME}" ws2_32)
endif()
include(CheckIncludeFiles)
include(CheckTypeSize)
include(CheckSymbolExists)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)

include(TestBigEndian)
include(FindBoost)
include(FindThreads)

include(FindCURL)
if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
#xxx FindCurl didn't create a target until CMake 3.12
    if(CURL_FOUND)
        if(NOT TARGET CURL::libcurl)
            add_library(CURL::libcurl UNKNOWN IMPORTED)
            set_target_properties(CURL::libcurl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}")
            set_property(TARGET CURL::libcurl APPEND PROPERTY IMPORTED_LOCATION "${CURL_LIBRARY}")
        endif()
    endif()
endif()


#include(FindZLIB)

# Configure compiler checks.
# These work in conjunction with the *_config.h.cmake.in files in the
#   various projects' include directories.
check_include_files("pthread.h"						HAVE_PTHREAD_H)
check_include_files("execinfo.h"					HAVE_EXECINFO_H)
check_symbol_exists("clock_gettime"		"time.h"	HAVE_CLOCK_GETTIME)
if (NOT HAVE_CLOCK_GETTIME) # On old systems this was in librt, not libc
	unset("HAVE_CLOCK_GETTIME" CACHE) # check_xxx_exists set CACHE variables, which cannot be re-used without being unset.
	check_library_exists(rt clock_gettime "time.h" 	HAVE_CLOCK_GETTIME)
	if (HAVE_CLOCK_GETTIME) # Record the necessary extra link library
		set(CLOCK_GETTIME_EXTRALIBS "rt" CACHE INTERNAL "")
	endif()
endif()
check_include_files("atomic.h"						HAVE_ATOMIC_H)
check_include_files("sys/time.h"					HAVE_SYS_TIME_H)
check_symbol_exists("localtime_r"		"time.h"	HAVE_LOCALTIME_R)
check_symbol_exists("gmtime_r"			"time.h"	HAVE_GMTIME_R)
check_symbol_exists("setenv"			"stdlib.h"	HAVE_SETENV)
check_symbol_exists("posix_memalign"	"stdlib.h"	HAVE_POSIX_MEMALIGN)
check_symbol_exists("memalign"			"stdlib.h"	HAVE_MEMALIGN)
test_big_endian(									BIGENDIAN)
check_type_size(	"size_t"						SIZEOF_SIZE_T)

#xxxTODO: Test this
find_package(Boost)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads)

find_package(CURL)
#xxx No longer needed?
#check_library_exists("curl" "curl_global_init" "" 	HAVE_CURL)  #xxx Need 'curl/curl.h'?

#find_package(ZLIB::ZLIB)

# Visual Studio 2013 has nullptr but not constexpr.  Need to check for
# both in here to make sure we have full C++11 support... otherwise,
# long-term we may need multiple separate configure checks and
# corresponding defines
#xxx This probably isn't the right way to test c++11; see cmake-compile-features()
check_cxx_source_compiles(
    "int main() { constexpr void* FOO = nullptr; }"	__CODA_CPP11)

check_symbol_exists("isnan"				"math.h"	HAVE_ISNAN)
# The auto-generated test code doesn't work for overloaded functions
check_cxx_source_compiles(
	"#include <cmath>\n    int main() { return std::isnan(0.0); }"
													HAVE_STD_ISNAN)

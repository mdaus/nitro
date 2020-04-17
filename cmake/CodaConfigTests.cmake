include(CheckCXXSourceCompiles)
include(CheckLibraryExists)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)
include(TestBigEndian)

# Configure compiler checks.
# These work in conjunction with the *_config.h.cmake.in files in the
#   various projects' include directories.
check_include_file("pthread.h" HAVE_PTHREAD_H)
check_include_file("execinfo.h" HAVE_EXECINFO_H)
check_symbol_exists("clock_gettime" "time.h" HAVE_CLOCK_GETTIME)
if (NOT HAVE_CLOCK_GETTIME) # On old systems this was in librt, not libc
    unset("HAVE_CLOCK_GETTIME" CACHE) # check_xxx_exists set CACHE variables, which cannot be re-used without being unset.
    find_library(RT_LIB rt)
    if (RT_LIB)
        check_library_exists(rt clock_gettime ${RT_LIB} HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME) # Record the necessary extra link library
            set(CLOCK_GETTIME_EXTRALIBS "rt" CACHE INTERNAL "")
        endif()
    endif()
endif()
check_include_file("atomic.h" HAVE_ATOMIC_H)
check_include_file("sys/time.h" HAVE_SYS_TIME_H)
check_symbol_exists("localtime_r" "time.h" HAVE_LOCALTIME_R)
check_symbol_exists("gmtime_r" "time.h" HAVE_GMTIME_R)
check_symbol_exists("setenv" "stdlib.h" HAVE_SETENV)
check_symbol_exists("posix_memalign" "stdlib.h" HAVE_POSIX_MEMALIGN)
check_symbol_exists("memalign" "stdlib.h" HAVE_MEMALIGN)
test_big_endian(BIGENDIAN)
check_type_size("size_t" SIZEOF_SIZE_T)


check_symbol_exists("isnan" "math.h" HAVE_ISNAN)
# The auto-generated test code doesn't work for overloaded functions
check_cxx_source_compiles("
    #include <cmath>
    int main() { return std::isnan(0.0); }
" HAVE_STD_ISNAN)

check_cxx_source_compiles("
    int __attribute__((noinline)) fn() { return 0; }
    int main() { return fn(); }
" HAVE_ATTRIBUTE_NOINLINE)

check_cxx_source_compiles("
    int main() { int var __attribute__((aligned (32))); return var; }
" HAVE_ATTRIBUTE_ALIGNED)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads)


find_package(CURL)
if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
    #FindCurl didn't create a target until CMake 3.12
    if(CURL_FOUND)
        if(NOT TARGET CURL::libcurl)
            add_library(CURL::libcurl UNKNOWN IMPORTED)
            set_target_properties(CURL::libcurl
                PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}")
            set_property(TARGET CURL::libcurl
                APPEND PROPERTY IMPORTED_LOCATION "${CURL_LIBRARY}")
        endif()
    endif()
endif()

set(BOOST_HOME "" CACHE PATH "path to boost installation")
if (BOOST_HOME)
    set(BOOST_ROOT ${BOOST_HOME})
endif()
# if boost serialization is found, the imported target Boost::serialization
# will be created to allow linking it
find_package(Boost COMPONENTS serialization)
set(HAVE_BOOST ${Boost_FOUND})


if (PYTHON_HOME)
    set(Python_ROOT_DIR ${PYTHON_HOME})
endif()
find_package(Python COMPONENTS Interpreter Development NumPy)
if (Python_FOUND AND Python_Development_FOUND)
    set(CODA_PYTHON_SITE_PACKAGES
        "${CODA_STD_PROJECT_LIB_DIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    if(NOT PYTHON_HOME)
        message("Python installation found at ${Python_EXECUTABLE}.\n"
                "Pass the configure option -DPYTHON_HOME=... to override this selection.")
    endif()
else()
    message(WARNING "Python targets will not be built since Python libraries were not found.\n"
            "Pass the configure option -DPYTHON_HOME=... to help locate an installation.")
endif()


find_package(SWIG)
if (SWIG_FOUND)
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
        cmake_policy(SET CMP0078 NEW) # UseSWIG generates standard target names
    endif()
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14")
        cmake_policy(SET CMP0086 NEW) # UseSWIG honors SWIG_MODULE_NAME via -module
    endif()
    include(${SWIG_USE_FILE})
else()
    message(WARNING "SWIG could not be found, so modules relying on it will not be built")
endif()

# sets OPENSSL_FOUND to TRUE if found, and creates targets
# OpenSSL:SSL and OpenSSL::Crypto
find_package(OpenSSL)

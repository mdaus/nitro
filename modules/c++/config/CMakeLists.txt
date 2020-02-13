set("TARGET_NAME"		"config")
set("TARGET_VERSION"	"1.0")

coda_add_library("${TARGET_NAME}")  # See the definition of this method for details on the variables it looks at.

include(CheckIncludeFile)
include(CheckSymbolExists)
include(TestBigEndian)
include(CheckTypeSize)
include(CheckCXXSourceCompiles)

check_include_file("pthread.h" HAVE_PTHREAD_H)
check_include_file("execinfo.h" HAVE_EXECINFO_H)
check_symbol_exists("clock_gettime" "time.h" HAVE_CLOCK_GETTIME)
check_include_file("atomic.h" HAVE_ATOMIC_H)
check_include_file("sys/time.h" HAVE_SYS_TIME_H)
check_symbol_exists("localtime_r" "time.h" HAVE_LOCALTIME_R)
check_symbol_exists("gmtime_r" "time.h" HAVE_GMTIME_R)
check_symbol_exists("setenv" "stdlib.h" HAVE_SETENV)
check_symbol_exists("posix_memalign" "stdlib.h" HAVE_POSIX_MEMALIGN)
check_symbol_exists("memalign" "stdlib.h" HAVE_MEMALIGN)
test_big_endian(BIGENDIAN)
check_type_size("size_t" SIZEOF_SIZE_T)

set(CODA_EXPORT " ")
if (MSVC AND BUILD_SHARED_LIBS)
    set(CODA_EXPORT "__declspec(dllexport)")
endif()

check_cxx_source_compiles("
    int __attribute__((noinline)) fn() { return 0; }
    int main()
    {
        return fn();
    }
" HAVE_ATTRIBUTE_NOINLINE)

check_cxx_source_compiles("
    int main()
    {
        int var __attribute__((aligned (32)));
        return var;
    }
" HAVE_ATTRIBUTE_ALIGNED)


set(CODA_CPP11 1)

configure_file(include/config/coda_oss_config.h.in include/config/coda_oss_config.h @ONLY)
target_include_directories("${TARGET_NAME}" INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}")

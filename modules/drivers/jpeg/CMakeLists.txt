set(TARGET_NAME jpeg)

option(ENABLE_JPEG "Enable use of libjpeg" ON)
set(JPEG_HOME "" CACHE STRING "path to libjpeg installation")

if (ENABLE_JPEG)
    if (JPEG_HOME STREQUAL "")
        message("JPEG_HOME not specified, so it will be built")

        set(DISABLE_DRIVER_BUILD ON) # custom build is used instead, below
        coda_add_driver(${TARGET_NAME}
            "jpegsrc.v9.tar"
            "SHA256=650de9b9c73b627c365c2c1617436bfdef47decddc1c6f7e006bd7655c4f8111"
        )
        set(HAVE_PROTOTYPES 1)
        check_cxx_source_compiles("int main() { unsigned short; return 0; }"
                                  HAVE_UNSIGNED_SHORT)
        check_cxx_source_compiles("int main() { unsigned char; return 0; }"
                                  HAVE_UNSIGNED_CHAR)
        check_include_file("stddef.h" HAVE_STDDEF_H)
        check_include_file("stdlib.h" HAVE_STDLIB_H)
        check_include_file("locale.h" HAVE_LOCALE_H)

        set(SOURCE_DIR "${${CMAKE_PROJECT_NAME}_${TARGET_NAME}_SOURCE_DIR}")
        foreach(src "cdjpeg.c"   "jaricom.c"  "jcapimin.c" "jcapistd.c"
                    "jcarith.c"  "jccoefct.c" "jccolor.c"  "jcdctmgr.c"
                    "jchuff.c"   "jcinit.c"   "jcmainct.c" "jcmarker.c"
                    "jcmaster.c" "jcomapi.c"  "jcparam.c"  "jcprepct.c"
                    "jcsample.c" "jctrans.c"  "jdapimin.c" "jdapistd.c"
                    "jdarith.c"  "jdatadst.c" "jdatasrc.c" "jdcoefct.c"
                    "jdcolor.c"  "jddctmgr.c" "jdhuff.c"   "jdinput.c"
                    "jdmainct.c" "jdmarker.c" "jdmaster.c" "jdmerge.c"
                    "jdpostct.c" "jdsample.c" "jdtrans.c"  "jerror.c"
                    "jfdctflt.c" "jfdctfst.c" "jfdctint.c" "jidctflt.c"
                    "jidctfst.c" "jidctint.c" "jmemmgr.c"  "jmemnobs.c"
                    "jquant1.c"  "jquant2.c"  "jutils.c"   "rdbmp.c"
                    "rdcolmap.c" "rdgif.c"    "rdppm.c"    "rdrle.c"
                    "rdswitch.c" "rdtarga.c"  "transupp.c" "wrbmp.c"
                    "wrgif.c"    "wrppm.c"    "wrrle.c"    "wrtarga.c")
            list(APPEND SOURCES "${SOURCE_DIR}/${src}")
        endforeach()

        set(JCONFIG "${${CMAKE_PROJECT_NAME}_${TARGET_NAME}_BINARY_DIR}/jconfig.h")
        configure_file("jconfig.h.in" ${JCONFIG})
        add_library(jpeg ${SOURCES})
        target_include_directories(${TARGET_NAME}
            PUBLIC "$<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>"
                   "$<BUILD_INTERFACE:${SOURCE_DIR}>"
                   "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_${TARGET_NAME}_BINARY_DIR}>")
        install(TARGETS jpeg
                LIBRARY DESTINATION "${CODA_STD_PROJECT_LIB_DIR}"
                ARCHIVE DESTINATION "${CODA_STD_PROJECT_LIB_DIR}")

        install(FILES "${SOURCE_DIR}/jpeglib.h"
                      "${SOURCE_DIR}/jmorecfg.h"
                      "${SOURCE_DIR}/jerror.h"
                      "${JCONFIG}"
                DESTINATION "${CODA_STD_PROJECT_INCLUDE_DIR}")
    else()
        set(frag "
            #include <stddef.h>
            #include <stdio.h>
            #include <jpeglib.h>

            int main(int argc, char **argv)
            {
                void *p;
                (void)argc; (void)argv;
                p=(void*)(jpeg_destroy_decompress);
                return 0;
            }")
        # TODO
    endif()
endif()



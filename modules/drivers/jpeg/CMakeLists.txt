set(TARGET_NAME "JPEG")

option(USE_LIBJPEG_TURBO "enable use of libjpeg-turbo instead of libjpeg" ON)

if (USE_LIBJPEG_TURBO)
    option(WITH_JPEG8 "use jpeg8 api" ON)
    option(WITH_TURBOJPEG "use turbojpeg api" OFF)

    # workaround for bug in libjpeg-turbo CMakeLists.txt
    set(CMAKE_INSTALL_DEFAULT_DATAROOTDIR "share")
    set(CMAKE_INSTALL_DEFAULT_DOCDIR "${CMAKE_INSTALL_DEFAULT_DATAROOTDIR}/doc")
    set(EXTRA_CMAKE_ARGS -DWITH_JPEG8=ON -DWITH_TURBOJPEG=OFF)
    coda_add_driver("${TARGET_NAME}"
        "libjpeg-turbo-2.0.4.tar.gz"
        "SHA256=33DD8547EFD5543639E890EFBF2EF52D5A21DF81FAF41BB940657AF916A23406"
    )
else()
    if (NOT WIN32)
        coda_add_driver("${TARGET_NAME}"
            "jpegsrc.v9.tar"
            "SHA256=650de9b9c73b627c365c2c1617436bfdef47decddc1c6f7e006bd7655c4f8111"
        )
    else()
        message(FATAL_ERROR "libjpeg not supported on Windows, use libjpeg-turbo instead")
    endif()
endif()

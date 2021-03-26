add_subdirectory("boost")
#add_subdirectory("curl") # this is handled in coda_find_system_dependencies
#add_subdirectory("numpy") # this is handled in coda_find_system_dependencies
#add_subdirectory("gsl") # nothing to do

set(ENABLE_J2K ON CACHE BOOL "enable J2K library")
set(J2K_HOME "" CACHE PATH "path to J2K installation")
if (ENABLE_J2K OR J2K_HOME)
    add_subdirectory("j2k")
endif()

set(ENABLE_JARS ON CACHE BOOL "include jars in the install")
if (ENABLE_JARS)
    add_subdirectory("jars")
endif()

set(ENABLE_JPEG ON CACHE BOOL "enable use of libjpeg")
set(JPEG_HOME "" CACHE PATH "path to libjpeg installation")
if (ENABLE_JPEG OR JPEG_HOME)
    add_subdirectory("jpeg")
endif()

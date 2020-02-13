# All directories under here contain C++ source
set("TARGET_LANG"		"c++")

#xxxTODO: Replace with a glob from all subdirectories once they're all working
add_subdirectory("config")
add_subdirectory("except")
add_subdirectory("str")
add_subdirectory("sys")
add_subdirectory("mem")
add_subdirectory("re")  # Can optionally use STD_REGEX
add_subdirectory("io")
add_subdirectory("zip")
add_subdirectory("sio.lite")
add_subdirectory("cli")
add_subdirectory("types")
add_subdirectory("unique")
add_subdirectory("math")
add_subdirectory("mt")
add_subdirectory("logging")
add_subdirectory("xml.lite")
add_subdirectory("net")
add_subdirectory("plugin")
add_subdirectory("tiff")

#Partially working
add_subdirectory("math.linear") # Needs Boost

# These ones aren't done yet.
## add_subdirectory("dbi")
## add_subdirectory("math.poly") # Breaking due to transitive include path problem with INTERFACE libs
## add_subdirectory("net.ssl")
## add_subdirectory("numpyutils")
## add_subdirectory("serialize")

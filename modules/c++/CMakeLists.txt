# All directories under here contain C++ source
set("TARGET_LANG"		"c++")

#xxxTODO: Replace with a glob from all subdirectories once they're all working
add_subdirectory("except")
add_subdirectory("str")
add_subdirectory("sys")
add_subdirectory("types")
add_subdirectory("math")
add_subdirectory("mem")
add_subdirectory("mt")
#add_subdirectory("cli")
#add_subdirectory("io")
#add_subdirectory("logging")
#add_subdirectory("plugin")
#add_subdirectory("sio.lite")
#add_subdirectory("tiff")

#Partially working
#add_subdirectory("math.linear")
#add_subdirectory("unique")
#add_subdirectory("re")
#add_subdirectory("net")

# These ones aren't done yet.
# add_subdirectory("dbi")
# add_subdirectory("math.poly") # Breaking due to transitive include path problem with INTERFACE libs
# add_subdirectory("net.ssl")
# add_subdirectory("numpyutils")
# add_subdirectory("serialize")
# add_subdirectory("xml.lite")
# add_subdirectory("zip")
# All directories under here contain C++ source
set(TARGET_LANG "c++")

add_subdirectory("config")
add_subdirectory("serialize")
add_subdirectory("except")
add_subdirectory("sys")
add_subdirectory("str")
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
add_subdirectory("net.ssl")
add_subdirectory("plugin")
add_subdirectory("tiff")
add_subdirectory("polygon")
add_subdirectory("math.linear") # Needs Boost for some tests
add_subdirectory("math.poly") # Needs Boost for some tests
add_subdirectory("numpyutils")

# These ones aren't done yet.
## add_subdirectory("dbi")

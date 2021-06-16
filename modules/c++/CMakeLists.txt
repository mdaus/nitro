set(TARGET_LANGUAGE c++)

# be sure we're really using C++20
if (MSVC)
    add_compile_options(/std:c++latest)
elseif (UNIX)
    add_compile_options(-std=c++2a)
endif()

add_subdirectory(nitf)

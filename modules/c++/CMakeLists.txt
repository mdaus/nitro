set(TARGET_LANGUAGE c++)

# turn on warnings as errors
if (MSVC)
    add_compile_options(/WX /W4) # /Wall
elseif (UNIX)
    add_compile_options(-std=c++14)

    #add_compile_options(-Werror -Wall -Wpedantic -Wextra)
    add_compile_options(-Werror -Wall -Wextra)
endif()

add_subdirectory(nitf)

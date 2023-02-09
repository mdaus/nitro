set(TARGET_LANGUAGE c++)

# turn on warnings as errors
if (MSVC)
    add_compile_options(/std:c++14)

    add_compile_options(/W4) # /Wall
    add_compile_options(/wd4996) # '...': This function or variable may be unsafe.
    add_compile_options(/wd4459) # declaration of '...' hides global declaration
elseif (UNIX)
    add_compile_options(-std=c++14)

    #add_compile_options(-Wall -pedantic -Wextra)
    add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(nitf)

set(TARGET_LANGUAGE c++)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
    add_compile_options(/W0)
elseif (UNIX)
    add_compile_options(-w) # "Inhibit all warning messages"
endif()

add_subdirectory("nitf")

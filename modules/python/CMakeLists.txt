if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
    add_compile_options(/W3)
    add_compile_options(/wd4267) # '...': conversion from '...' to '...', possible loss of data
    add_compile_options(/wd4244) # '...': conversion from '...' to '...', possible loss of data
elseif (UNIX)
    add_compile_options(-Wno-sign-compare)
    add_compile_options(-Wno-class-memaccess)
    add_compile_options(-Wno-unused-function -Wno-unused-variable)
endif()

add_subdirectory("nitf")

set(CMAKE_C_STANDARD 90)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (UNIX)
    add_compile_options(-Wno-deprecated -Wno-deprecated-declarations)
elseif(MSVC)
    add_definitions(-DWIN32_LEAN_AND_MEAN
                    -DNOMINMAX
                    -D_CRT_SECURE_NO_WARNINGS
                    -D_SCL_SECURE_NO_WARNINGS
                    -D_USE_MATH_DEFINES)
endif()

function(import_coda_module module)
    add_library(${module} STATIC IMPORTED)
    if(UNIX)
        set_target_properties(${module} PROPERTIES
            IMPORTED_LOCATION ${CODA_INSTALL_DIR}/lib/lib${module}.a)
    elseif(MSVC)
        set_target_properties(${module} PROPERTIES
            IMPORTED_LOCATION ${CODA_INSTALL_DIR}/lib/${module}.lib)
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()
endfunction()


# this file contains common functions and macros for initializing and adding
# components to the build

set(CODA_OSS_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL
    "path to coda-oss directory")

# Set up the global build configuration
macro(coda_initialize_build)
    option(CODA_PARTIAL_INSTALL "Allow building and installing a subset of the targets" OFF)
    if (CODA_PARTIAL_INSTALL)
        # This setting, along with setting all install commands as "OPTIONAL",
        # allows installing a subset of the targets.
        #
        # CMake's default forces everything to be built before the install target is
        # run. This override allows one to build a subset of the project and then
        # run the install target to install only what has been built (the build and
        # install must be run as separate steps). For example:
        #
        #   cmake --build . --target target1
        #   cmake --build . --target target2
        #   cmake --build . --target install
        #
        # or in CMake 3.15+
        #   cmake --build . --target target1 target2
        #   cmake --build . --target install
        #
        # to build and install everything, run (with Unix Makefiles)
        #   cmake --build . --target all
        #   cmake --build . --target install
        #
        # and for MSVC:
        #   cmake --build . --target ALL_BUILD
        #   cmake --build . --target install
        #
        # This feature still has some rough edges, in that files and directories
        # that are installed from the source tree are always installed (because they
        # always exist regardless of what was built). This could be fixed by copying
        # them from the source tree to the build directory during the build stage
        # and then pointing the install commands at the file paths within the build
        # directory.
        set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)
        set(CODA_INSTALL_OPTION OPTIONAL)
    endif()

    # Standard directory names used throughout the project.
    set(CODA_STD_PROJECT_INCLUDE_DIR    "include")
    set(CODA_STD_PROJECT_LIB_DIR        "lib")
    set(CODA_STD_PROJECT_BIN_DIR        "bin")

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Detect 32/64-bit architecture
    if(NOT CODA_BUILD_BITSIZE)
        if (CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(CODA_BUILD_BITSIZE "64" CACHE STRING "Select Architecture" FORCE)
        elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(CODA_BUILD_BITSIZE "32" CACHE STRING "Select Architecture" FORCE)
        else()
            message(FATAL_ERROR "Unknown Pointer Size: ${CMAKE_SIZEOF_VOID_P} Bytes")
        endif()
        set_property(CACHE CODA_BUILD_BITSIZE PROPERTY STRINGS "64" "32")
    endif()

    if (MSVC)
        # MSVC is a multi-configuration generator, which typically allows build
        # type to be selected at build time rather than configure time. However,
        # we don't support that currently as it will take some work to be able
        # to get the install step working with multiple configurations.
        set(CMAKE_BUILD_TYPE "" CACHE STRING "Select Build Type")
        if((NOT CMAKE_BUILD_TYPE STREQUAL "Release") AND
           (NOT CMAKE_BUILD_TYPE STREQUAL "Debug") AND
           (NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo") AND
           (NOT CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
            message(FATAL_ERROR "must specify CMAKE_BUILD_TYPE as one of [Release, Debug, RelWithDebInfo, MinSizeRel]")
        endif()
    else()
        set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Select Build Type" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

    option(BUILD_SHARED_LIBS "Build shared libraries instead of static." OFF)
    if(BUILD_SHARED_LIBS)
        set(CODA_LIBRARY_TYPE "shared")
    else()
        set(CODA_LIBRARY_TYPE "static")
    endif()

    include("${CODA_OSS_SOURCE_DIR}/build/config_tests.cmake") # Code to test compiler features

    option(CODA_BUILD_TESTS "build tests" ON)
    if (CODA_BUILD_TESTS)
        enable_testing()
    endif()

    # Set default install directory
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX
            "${PROJECT_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME}${CODA_BUILD_BITSIZE}-${CMAKE_CXX_COMPILER_ID}-${CODA_LIBRARY_TYPE}-${CMAKE_BUILD_TYPE}"
            CACHE PATH "Install directory" FORCE)
        message("CMAKE_INSTALL_PREFIX was not specified, defaulting to ${CMAKE_INSTALL_PREFIX}")
    endif()

    # Look for things in our own install location first.
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}")

    # MSVC-specific flags and options.
    if (MSVC)
        set_property(GLOBAL PROPERTY USE_FOLDERS ON)

        # change default dynamic linked CRT (/MD or /MDd) to static linked CRT
        # (/MT or /MTd) if requested
        option(STATIC_CRT "use static CRT library /MT (or /MTd for Debug)" OFF)
        foreach(build_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_${build_config} "${CMAKE_CXX_FLAGS_${build_config}}")
            string(REGEX REPLACE "/MD" "/MT" CMAKE_C_FLAGS_${build_config} "${CMAKE_C_FLAGS_${build_config}}")
        endforeach()

        # set default warning level to /W3
        string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/W[0-4]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        add_compile_options(/W3)

        # catch exceptions that bubble through a C-linkage layer
        string(REGEX REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        add_compile_options(/EHs)

        add_definitions(
            -DWIN32_LEAN_AND_MEAN
            -DNOMINMAX
            -D_CRT_SECURE_NO_WARNINGS
            -D_SCL_SECURE_NO_WARNINGS
            -D_USE_MATH_DEFINES
        )
        add_compile_options(
            /wd4290
            /wd4512
            #/MP        # Parallel Build
        )

        link_libraries(    # CMake uses this for both libraries and linker options
            -STACK:80000000
        )

        # This should probably be replaced by GenerateExportHeader
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD TRUE)
    endif()

    # Unix/Linux specific options
    if (UNIX)
        add_definitions(
            -D_LARGEFILE_SOURCE
            -D_FILE_OFFSET_BITS=64
        )
        add_compile_options(
            -Wno-deprecated
            -Wno-unused-value
            -Wno-unused-but-set-variable
        )
    endif()

    if (SWIG_FOUND AND Python_FOUND AND Python_Development_FOUND)
        option(PYTHON_EXTRA_NATIVE "generate extra native containers with SWIG" ON)
        set(CMAKE_SWIG_FLAGS "")
        if (PYTHON_EXTRA_NATIVE)
            list(APPEND CMAKE_SWIG_FLAGS "-extranative")
        endif()
        if (Python_VERSION_MAJOR GREATER_EQUAL 3)
            list(APPEND CMAKE_SWIG_FLAGS "-py3")
        endif()
    endif()

    # show the compile options for the project directory
    get_property(tmp_compile_options DIRECTORY "./" PROPERTY COMPILE_OPTIONS)
    message("COMPILE OPTIONS=${tmp_compile_options}")
    unset(tmp_compile_options)

    message("CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
    message("CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
    message("CMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}")
    message("CMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}")
    message("CMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    message("CMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    message("CMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")
    message("CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}")
    message("CMAKE_SWIG_FLAGS=${CMAKE_SWIG_FLAGS}")
endmacro()


# Utility to filter a list of files.
#
# dest_name     - Destination variable name in parent's scope
# file_list     - Input list of files (possibly with paths)
# filter_list   - Input list of files to filter out
#                   (must be bare filenames; no paths)
#
function(filter_files dest_name file_list filter_list)
    foreach(test_src ${file_list})
        get_filename_component(test_src_name ${test_src} NAME)
        if (NOT ${test_src_name} IN_LIST filter_list)
            list(APPEND good_names ${test_src})
        endif()
    endforeach()
    set(${dest_name} ${good_names} PARENT_SCOPE)
endfunction()


# Add a driver (3rd-party library) to the build.
#
# Single value arguments:
#   NAME        - Name of the driver
#   ARCHIVE     - Location of the archive containing the driver.
#                 This can be a path relative to ${CMAKE_CURRENT_SOURCE_DIR}.
#   HASH        - hash signature in the form hashtype=hashvalue
#
#  The 3P's source and build directories will be stored in
#       ${${target_name_lc}_SOURCE_DIR}, and
#       ${${target_name_lc}_BINARY_DIR} respectively,
#
#       where ${target_name_lc} is the lower-cased target name.
#
include(FetchContent) # Requires CMake 3.11+
include(ExternalProject)
function(coda_add_driver)
    cmake_parse_arguments(
        ARG                          # prefix
        "BUILD"                      # options
        "NAME;ARCHIVE;HASH"          # single args
        ""                           # multi args
        "${ARGN}"
    )
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "received unexpected argument(s): ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    set(target_name ${CMAKE_PROJECT_NAME}_${ARG_NAME})
    # Use 'FetchContent' to download and unpack the files.  Set it up here.
    FetchContent_Declare(${target_name}
        URL "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_ARCHIVE}"
        URL_HASH ${ARG_HASH}
    )
    FetchContent_GetProperties(${target_name})
    # The returned properties use the lower-cased name
    string(TOLOWER ${target_name} target_name_lc)
    if (NOT ${target_name_lc}_POPULATED) # This makes sure we only fetch once.
        message("Populating content for external dependency ${driver_name}")
        # Now (at configure time) unpack the content.
        FetchContent_Populate(${target_name})
        # Remember where we put stuff
        set("${target_name_lc}_SOURCE_DIR" "${${target_name_lc}_SOURCE_DIR}"
            CACHE INTERNAL "source directory for ${target_name_lc}")
        set("${target_name_lc}_BINARY_DIR" "${${target_name_lc}_BINARY_DIR}"
            CACHE INTERNAL "binary directory for ${target_name_lc}")
    endif()
endfunction()


# Add a module's tests or unit tests to the build
#
# Single value arguments:
#   MODULE_NAME     - Name of the module
#   DIRECTORY       - Directory containing the tests' source code, relative to
#                     the current source directory. All source files beneath
#                     this directory will be used. Each source file is assumed
#                     to create a separate executable.
#
# Multi value arguments:
#   DEPS            - Modules that the tests are dependent upon.
#   FILTER_LIST     - Source files to ignore
#
# Option arguments:
#   UNITTEST        - If present, the test will be added to the CTest suite for
#                     automated running.
#
function(coda_add_tests)
    if (CODA_BUILD_TESTS)
        cmake_parse_arguments(
            ARG                         # prefix
            "UNITTEST"                  # options
            "MODULE_NAME;DIRECTORY"     # single args
            "DEPS;FILTER_LIST"          # multi args
            "${ARGN}"
        )
        if (ARG_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "received unexpected argument(s): ${ARG_UNPARSED_ARGUMENTS}")
        endif()

        if (NOT ARG_DIRECTORY)
            message(FATAL_ERROR "Must give a test directory")
        endif()
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_DIRECTORY}")
            message(FATAL_ERROR "Directory ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_DIRECTORY} does not exist")
        endif()
        # Find all the source files, relative to the module's directory
        file(GLOB_RECURSE local_tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
             "${ARG_DIRECTORY}/*.cpp")
        # Filter out ignored files
        filter_files(local_tests "${local_tests}" "${ARG_FILTER_LIST}")

        # make a group target to build all tests for the current module
        set(test_group_tgt "${ARG_MODULE_NAME}_tests")
        if (NOT TARGET ${test_group_tgt})
            add_custom_target(${test_group_tgt})
        endif()

        list(APPEND ARG_DEPS ${ARG_MODULE_NAME})

        set(module_dep_targets TestCase)
        foreach(dep ${ARG_DEPS})
            if (NOT ${dep} STREQUAL "")
                list(APPEND module_dep_targets "${dep}-c++")
            endif()
        endforeach()

        # get all interface libraries and include directories from the dependencies
        foreach(dep ${module_dep_targets})
            if (TARGET ${dep})
                get_property(dep_includes TARGET ${dep}
                             PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
                list(APPEND include_dirs ${dep_includes})
            endif()
        endforeach()

        foreach(test_src ${local_tests})
            # Use the base name of the source file as the name of the test
            get_filename_component(test_name "${test_src}" NAME_WE)
            set(test_target "${ARG_MODULE_NAME}_${test_name}")
            add_executable(${test_target} "${test_src}")
            set_target_properties(${test_target} PROPERTIES OUTPUT_NAME ${test_name})
            add_dependencies(${test_group_tgt} ${test_target})
            get_filename_component(test_dir "${test_src}" DIRECTORY)
            # Do a bit of path manipulation to make sure tests in deeper subdirs retain those subdirs in their build outputs
            file(RELATIVE_PATH test_subdir
                 "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_DIRECTORY}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}")
            # message(STATUS "Generating Test: module_name=${ARG_MODULE_NAME} test_src=${test_src}  test_name=${test_name}  test_dir=${test_dir}  test_subdir= ${test_subdir} module_deps=${ARG_DEPS}")

            # Set IDE subfolder so that tests appear in their own tree
            set_target_properties(${test_target}
                PROPERTIES FOLDER "${ARG_DIRECTORY}/${ARG_MODULE_NAME}/${test_subdir}")

            target_link_libraries(${test_target} PRIVATE ${module_dep_targets})
            target_include_directories(${test_target} PRIVATE ${include_dirs})

            # add unit tests to automatic test suite
            if (${ARG_UNITTEST})
                add_test(NAME ${test_target} COMMAND ${test_target})
            endif()

            # Install [unit]tests to separate subtrees
            install(TARGETS ${test_target} ${CODA_INSTALL_OPTION}
                    RUNTIME DESTINATION "${ARG_DIRECTORY}/${ARG_MODULE_NAME}/${test_subdir}")
        endforeach()
    endif()
endfunction()


# Generate a configuration header file for a coda module.
#
# Arguments:
#   MODULE_NAME     - Name of the module to which the configuration file belongs
#
function(coda_generate_module_config_header MODULE_NAME)
    # Periods in target names for dirs are replaced with slashes (subdirectories).
    string(REPLACE "." "/" tgt_munged_dirname ${MODULE_NAME})

    # Periods in target names for files are replaced with underscores.
    # Note that this variable name is used in the *.cmake.in files.
    string(REPLACE "." "_" tgt_munged_name ${MODULE_NAME})

    # If we find a *_config.h.cmake.in file, generate the corresponding *_config.h, and put the
    #   target directory in the include path.
    set(config_file_template "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${MODULE_NAME}_config.h.cmake.in")
    if (EXISTS ${config_file_template})
        set(config_file_out "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}/${tgt_munged_name}_config.h")
        message(STATUS "Processing config header: ${config_file_template} -> ${config_file_out}")
        configure_file("${config_file_template}" "${config_file_out}")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${config_file_out}"
                DESTINATION "${CODA_STD_PROJECT_INCLUDE_DIR}/${tgt_munged_dirname}"
                ${CODA_INSTALL_OPTION})
    endif()
endfunction()


# Add a C++ module to the build
#
# Single value arguments:
#   MODULE_NAME     - Name of the module
#
# Multi value arguments:
#   DEPS            - List of internal module dependencies for the library
#   EXTERNAL_DEPS   - List of linkable external dependencies for the library
#   EXTRA_DEPS      - List of non-linkable dependencies for the library
#   SOURCE_FILTER   - Source files to ignore
#
function(coda_add_module)
    cmake_parse_arguments(
        ARG                                             # prefix
        ""                                              # options
        "NAME;VERSION"                                  # single args
        "DEPS;EXTERNAL_DEPS;EXTRA_DEPS;SOURCE_FILTER"   # multi args
        "${ARGN}"
    )
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "received unexpected argument(s): ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    set(target_name "${ARG_NAME}-c++")

    # Find all the source files, relative to the module's directory
    file(GLOB_RECURSE local_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "source/*.cpp")

    if (GPU_ENABLED AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
        file(GLOB_RECURSE cuda_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "kernels/*.cu")
        list(APPEND local_sources ${cuda_sources})
    endif()

    # Filter out ignored files
    filter_files(local_sources "${local_sources}" "${ARG_SOURCE_FILTER}")

    coda_generate_module_config_header(${ARG_NAME})

    if (NOT local_sources)
        # Libraries without sources must be declared to CMake as INTERFACE libraries
        set(lib_type INTERFACE)
        add_library(${target_name} INTERFACE)
    else()
        set(lib_type PUBLIC)
        add_library(${target_name} ${local_sources})
    endif()

    # convert module dependency names to target names (append "-c++")
    foreach(dep ${ARG_DEPS})
        list(APPEND module_dep_targets "${dep}-c++")
    endforeach()

    target_link_libraries(${target_name} ${lib_type}
        ${module_dep_targets}
        ${ARG_EXTERNAL_DEPS})

    target_include_directories(${target_name} ${lib_type}
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>")

    if (ARG_EXTRA_DEPS)
        add_dependencies(${target_name} ${ARG_EXTRA_DEPS})
    endif()

    if (cuda_sources)
        get_property(tmp TARGET ${target_name} PROPERTY COMPILE_OPTIONS)
        set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            COMPILE_OPTIONS $<IF:$<STREQUAL:$<COMPILE_LANGUAGE>,CUDA>,-rdc=true,>
        )
    endif()

    # Set up install destinations for binaries
    install(TARGETS ${target_name}
            EXPORT "${ARG_NAME}_TARGETS"
            ${CODA_INSTALL_OPTION}
            LIBRARY DESTINATION "${CODA_STD_PROJECT_LIB_DIR}"
            ARCHIVE DESTINATION "${CODA_STD_PROJECT_LIB_DIR}"
            RUNTIME DESTINATION "${CODA_STD_PROJECT_BIN_DIR}")

    # Set up install destination for headers
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}"
            DESTINATION "."
            ${CODA_INSTALL_OPTION}
            FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.hpp")

    # install conf directory, if present
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conf")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/conf"
                DESTINATION "share/${ARG_NAME}"
                ${CODA_INSTALL_OPTION})
    endif()

    # cannot use exports until all external dependencies have their own exports defined
    #install(EXPORT "${target_name}_TARGETS"
    #    FILE ${target_name}_TARGETS.cmake
    #    NAMESPACE ${target_name}::
    #    DESTINATION ${CODA_STD_PROJECT_LIB_DIR}/cmake/${target_name}
    #)

#[[  #xxx TODO Export the library interface? See https://www.youtube.com/watch?v=bsXLMQ6WgIk
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        VERSION ${${target_name}_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES "${tgt_munged_name}_config.cmake" "${tgt_munged_name}_config_version.cmake"
        DESTINATION ${CODA_STD_PROJECT_LIB_DIR/cmake/${tgt_munged_dirname}
    )
    # Then, pull in:
    include(CMakeFindDependencyMacro)
    find_dependency(mydepend 1.0) # Version#
    include("${CMAKE_CURRENT_LIST_DIR}/${tgt_munged_name}_TARGETS.cmake")

    #xxx Also, add_library("${target_name}::${target_name}" ALIAS ${target_name})
#]]
endfunction()


# Add a plugin (dynamically loaded library) to the build
#
# Single value arguments:
#   PLUGIN_NAME     - Name of the plugin
#   PLUGIN          - The module to which this plugin attaches
#   VERSION         - Version number of this plugin
#
# Multi value arguments:
#   MODULE_DEPS     - List of module dependencies for the plugin
#   SOURCES         - Optional list of source files for compiling the plugin.
#                     If not provided, the source files will be globbed.
#
function(coda_add_plugin)
    cmake_parse_arguments(
        ARG                          # prefix
        ""                           # options
        "PLUGIN_NAME;PLUGIN;VERSION" # single args
        "MODULE_DEPS;SOURCES"        # multi args
        "${ARGN}"
    )
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "received unexpected argument(s): ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    set(OUTPUT_NAME "${ARG_PLUGIN_NAME}-c++")
    set(TARGET_NAME "${ARG_PLUGIN}_${OUTPUT_NAME}")

    if (NOT ARG_SOURCES)
        file(GLOB SOURCES "source/*.cpp")
    else()
        set(SOURCES "${ARG_SOURCES}")
    endif()

    add_library(${TARGET_NAME} MODULE "${SOURCES}")
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_include_directories(${TARGET_NAME}
            PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                   "$<INSTALL_INTERFACE:${CODA_STD_PROJECT_INCLUDE_DIR}>")
    endif()

    foreach(dep ${ARG_MODULE_DEPS})
        list(APPEND MODULE_DEP_TARGETS "${dep}-c++")
    endforeach()
    target_link_libraries(${TARGET_NAME} PUBLIC ${MODULE_DEP_TARGETS})

    target_compile_definitions(${TARGET_NAME} PRIVATE PLUGIN_MODULE_EXPORTS)

    install(TARGETS ${TARGET_NAME}
            ${CODA_INSTALL_OPTION}
            LIBRARY DESTINATION "share/${ARG_PLUGIN}/plugins"
            ARCHIVE DESTINATION "share/${ARG_PLUGIN}/plugins"
            RUNTIME DESTINATION "share/${ARG_PLUGIN}/plugins")

    # install headers
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${CODA_STD_PROJECT_INCLUDE_DIR}"
            DESTINATION "."
            FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.hpp"
            ${CODA_INSTALL_OPTION})

    # install conf directory, if present
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conf")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/conf"
                DESTINATION "share/${ARG_PLUGIN}"
                ${CODA_INSTALL_OPTION})
    endif()
endfunction()


# Add a SWIG Python module to the build
#
# Single value arguments:
#   TARGET          - Name of the CMake target to build the Python module
#   MODULE_NAME     - Name of the module within Python
#   INPUT           - Source file (.i) from which to generate the SWIG bindings
#   PACKAGE         - Name of the package to which this module is added
#
# Multi value arguments:
#   DEPS            - List of compiled module dependencies for the library
#   PYTHON_DEPS     - List of Python module dependencies for the library
#
function(coda_add_swig_python_module)
    cmake_parse_arguments(
        "ARG"                               # prefix
        ""                                  # options
        "TARGET;MODULE_NAME;INPUT;PACKAGE"  # single args
        "MODULE_DEPS;PYTHON_DEPS"           # multi args
        "${ARGN}"
    )
    if (NOT ARG_PACKAGE)
        message(FATAL_ERROR "package must be specified for Python module ${ARG_TARGET}")
    endif()

    # determine the necessary includes from the compiled dependencies
    set(include_dirs "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source>")
    set(libs ${Python_LIBRARIES})
    foreach(dep ${ARG_MODULE_DEPS})
        get_property(dep_interface_include_dirs TARGET ${dep}
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(APPEND include_dirs ${dep_interface_include_dirs})
    endforeach()

    # get SWIG include directories from the Python dependencies
    foreach(dep ${ARG_PYTHON_DEPS})
        get_property(dep_swig_include_dirs TARGET ${dep}
                     PROPERTY SWIG_INCLUDE_DIRECTORIES)
        list(APPEND include_dirs ${dep_swig_include_dirs})
    endforeach()

    set_property(SOURCE ${ARG_INPUT} PROPERTY CPLUSPLUS ON)
    set_property(SOURCE ${ARG_INPUT} PROPERTY SWIG_MODULE_NAME ${ARG_MODULE_NAME})
    set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/source/generated")
    set(SWIG_OUTFILE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source/generated")

    swig_add_library(${ARG_TARGET} LANGUAGE python SOURCES ${ARG_INPUT})

    target_link_libraries(${ARG_TARGET} PRIVATE "${ARG_MODULE_DEPS}")
    set_property(TARGET ${ARG_TARGET} PROPERTY
        SWIG_INCLUDE_DIRECTORIES "${include_dirs}")
    set_property(TARGET ${ARG_TARGET} PROPERTY
        SWIG_GENERATED_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}")
    set_property(TARGET ${ARG_TARGET} PROPERTY
        LIBRARY_OUTPUT_NAME ${ARG_MODULE_NAME})
    file(GLOB sources_py
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/generated/*.py")

    # install the compiled extension library
    install(TARGETS ${ARG_TARGET}
            DESTINATION "${CODA_PYTHON_SITE_PACKAGES}/${ARG_PACKAGE}"
            ${CODA_INSTALL_OPTION})

    # install the python scripts
    install(FILES ${sources_py}
            DESTINATION "${CODA_PYTHON_SITE_PACKAGES}/${ARG_PACKAGE}"
            ${CODA_INSTALL_OPTION})
endfunction()
